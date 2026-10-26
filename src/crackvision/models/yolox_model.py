"""YOLOX : construction du reseau et inference.

Le reseau est celui de la bibliotheque officielle (Megvii, Apache 2.0) ; seul le
nombre de classes change (1 : fissure). Les variantes reprennent les facteurs de
profondeur et de largeur des fichiers `exps/default/yolox_*.py`.

Convention YOLOX : image BGR, valeurs 0-255 non normalisees, redimensionnement
avec conservation du ratio et remplissage gris (114) en bas et a droite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from crackvision.utils import read_json


@dataclass(frozen=True)
class Variant:
    depth: float
    width: float
    depthwise: bool = False


VARIANTS = {
    "nano": Variant(0.33, 0.25, depthwise=True),
    "tiny": Variant(0.33, 0.375),
    "s": Variant(0.33, 0.50),
    "m": Variant(0.67, 0.75),
    "l": Variant(1.0, 1.0),
}


def build_model(variant: str, num_classes: int = 1, act: str = "silu") -> nn.Module:
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    cfg = VARIANTS[variant]
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(cfg.depth, cfg.width, in_channels=in_channels, act=act,
                         depthwise=cfg.depthwise)
    head = YOLOXHead(num_classes, cfg.width, in_channels=in_channels, act=act,
                     depthwise=cfg.depthwise)
    model = YOLOX(backbone, head)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    model.head.initialize_biases(1e-2)
    return model


def preprocess(image_bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Equivalent de yolox.data.data_augment.preproc (sans importer yolox.data)."""
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / image_bgr.shape[0], input_size[1] / image_bgr.shape[1])
    resized = cv2.resize(image_bgr, (int(image_bgr.shape[1] * ratio),
                                     int(image_bgr.shape[0] * ratio)),
                         interpolation=cv2.INTER_LINEAR)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


class YoloxDetector:
    """Detecteur pret a l'emploi : liste d'images BGR -> boites (x1, y1, x2, y2, score)."""

    def __init__(self, model: nn.Module, input_size: int, nms: float = 0.5,
                 num_classes: int = 1, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.input_size = (int(input_size), int(input_size))
        self.nms = float(nms)
        self.num_classes = num_classes
        self.device = device

    @classmethod
    def from_dir(cls, model_dir: str | Path, device: str = "cpu") -> YoloxDetector:
        model_dir = Path(model_dir)
        meta = read_json(model_dir / "meta.json")
        model = build_model(meta["variant"], num_classes=int(meta["num_classes"]))
        state = torch.load(model_dir / "model.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(model, meta["input_size"], meta["nms"], int(meta["num_classes"]), device)

    @torch.no_grad()
    def predict(self, images: list[np.ndarray], conf: float = 0.01,
                batch_size: int = 16) -> list[np.ndarray]:
        from yolox.utils import postprocess

        results: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            arrays, ratios = zip(*(preprocess(img, self.input_size) for img in batch),
                                 strict=True)
            tensor = torch.from_numpy(np.stack(arrays)).to(self.device)
            outputs = self.model(tensor)
            outputs = postprocess(outputs, self.num_classes, conf, self.nms,
                                  class_agnostic=True)
            for image, output, ratio in zip(batch, outputs, ratios, strict=True):
                if output is None:
                    results.append(np.zeros((0, 5), dtype=np.float32))
                    continue
                output = output.float().cpu().numpy()
                boxes = output[:, :4] / ratio
                boxes[:, 0::2] = boxes[:, 0::2].clip(0, image.shape[1])
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, image.shape[0])
                scores = output[:, 4] * output[:, 5]
                results.append(np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32))
        return results
