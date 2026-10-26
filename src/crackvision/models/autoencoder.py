"""Autoencodeur convolutif entraine uniquement sur des tuiles saines.

Principe : appris sur du beton sain, le reseau sait reconstruire la texture
normale mais pas une fissure. L'ecart entre l'image et sa reconstruction donne
une carte d'anomalie ; son seuil se calibre sur des tuiles saines de validation.

Carte d'anomalie = (1 - w) * |x - x_rec| + w * (1 - SSIM(x, x_rec)) / 2, lissee
par un filtre gaussien. Le terme SSIM (Bergmann et al., 2019) compare la
structure locale plutot que l'intensite pixel a pixel : il reagit moins au grain
du beton et davantage aux discontinuites comme une fissure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from crackvision.utils import read_json


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4,
                 latent_channels: int = 32):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for level in range(depth):
            out = base_channels * 2 ** level
            layers += [_conv_block(channels, out, stride=2), _conv_block(out, out)]
            channels = out
        # Goulot d'etranglement : 1x1 vers peu de canaux, a 1/2^depth de la resolution
        layers += [nn.Conv2d(channels, latent_channels, 1), nn.BatchNorm2d(latent_channels),
                   nn.LeakyReLU(0.2, inplace=True)]
        self.encoder = nn.Sequential(*layers)

        layers = [_conv_block(latent_channels, channels)]
        for level in reversed(range(depth)):
            out = base_channels * 2 ** (level - 1) if level > 0 else base_channels
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       _conv_block(channels, out), _conv_block(out, out)]
            channels = out
        layers += [nn.Conv2d(channels, in_channels, 3, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- SSIM
def _gaussian_kernel(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def _filter(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=x.shape[1])


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Carte SSIM moyennee sur les canaux : (B, 1, H, W), valeurs dans [-1, 1]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    kernel = _gaussian_kernel(window, sigma, x.shape[1], x.device, x.dtype)
    mu_x, mu_y = _filter(x, kernel), _filter(y, kernel)
    sigma_x = _filter(x * x, kernel) - mu_x ** 2
    sigma_y = _filter(y * y, kernel) - mu_y ** 2
    sigma_xy = _filter(x * y, kernel) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.mean(dim=1, keepdim=True)


def reconstruction_loss(x: torch.Tensor, recon: torch.Tensor, ssim_weight: float) -> torch.Tensor:
    l1 = (x - recon).abs().mean()
    structural = 1 - ssim_map(x, recon).mean()
    return (1 - ssim_weight) * l1 + ssim_weight * structural


def anomaly_maps(x: torch.Tensor, recon: torch.Tensor, ssim_weight: float,
                 blur_sigma: float) -> torch.Tensor:
    """Cartes d'anomalie (B, H, W)."""
    l1 = (x - recon).abs().mean(dim=1, keepdim=True)
    structural = ((1 - ssim_map(x, recon)) / 2).clamp(0, 1)
    amap = (1 - ssim_weight) * l1 + ssim_weight * structural
    if blur_sigma > 0:
        size = int(2 * round(3 * blur_sigma) + 1)
        amap = _filter(amap, _gaussian_kernel(size, blur_sigma, 1, amap.device, amap.dtype))
    return amap[:, 0]


def image_scores(maps: torch.Tensor, topk: float) -> torch.Tensor:
    """Score d'image = moyenne des k % de pixels les plus anormaux."""
    flat = maps.flatten(start_dim=1)
    count = max(1, int(round(flat.shape[1] * topk)))
    return flat.topk(count, dim=1).values.mean(dim=1)


# --------------------------------------------------------------------------- inference
def to_tensor(images_bgr: list[np.ndarray], size: int) -> torch.Tensor:
    arrays = []
    for image in images_bgr:
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        arrays.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(arrays)).float() / 255.0


class AnomalyDetector:
    """Autoencodeur pret a l'emploi : images BGR -> cartes d'anomalie et scores."""

    def __init__(self, model: nn.Module, meta: dict, calibration: dict | None = None,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.meta = meta
        self.calibration = calibration or {}
        self.device = device

    @classmethod
    def from_dir(cls, model_dir: str | Path, device: str = "cpu") -> AnomalyDetector:
        model_dir = Path(model_dir)
        meta = read_json(model_dir / "meta.json")
        model = ConvAutoencoder(base_channels=meta["base_channels"], depth=meta["depth"],
                                latent_channels=meta["latent_channels"])
        model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu",
                                         weights_only=True))
        calibration_path = model_dir / "calibration.json"
        calibration = read_json(calibration_path) if calibration_path.exists() else None
        return cls(model, meta, calibration, device)

    @torch.no_grad()
    def predict(self, images: list[np.ndarray], batch_size: int = 32,
                return_reconstruction: bool = False):
        """Retourne (cartes a la taille de chaque image, scores[, reconstructions])."""
        size = int(self.meta["image_size"])
        maps, scores, recons = [], [], []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            x = to_tensor(batch, size).to(self.device)
            recon = self.model(x)
            amap = anomaly_maps(x, recon, self.meta["ssim_weight"], self.meta["blur_sigma"])
            scores.extend(image_scores(amap, self.meta["score_topk"]).cpu().numpy().tolist())
            for image, small in zip(batch, amap.cpu().numpy(), strict=True):
                maps.append(cv2.resize(small, (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_LINEAR))
            if return_reconstruction:
                recon_np = (recon.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                recons.extend(cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in recon_np)
        scores_np = np.asarray(scores, dtype=np.float32)
        if return_reconstruction:
            return maps, scores_np, recons
        return maps, scores_np

    def score_z(self, scores: np.ndarray) -> np.ndarray:
        """Score centre-reduit par rapport aux tuiles saines de validation."""
        mean = self.calibration.get("score_mean", 0.0)
        std = max(self.calibration.get("score_std", 1.0), 1e-6)
        return (np.asarray(scores) - mean) / std

    @property
    def pixel_threshold(self) -> float:
        return float(self.calibration.get("pixel_threshold", 1.0))
