"""Etape `train_autoencoder` : autoencodeur debruiteur sur tuiles saines.

- entrainement : tuiles saines du jeu train uniquement, augmentations qui
  preservent la normalite (symetries, rotations de 90 degres, legere variation
  d'eclairage) et bruit gaussien en entree ;
- choix du point de controle sur l'AUROC de la validation annotee (ou, au choix,
  sur la perte de reconstruction des tuiles saines de validation) ;
- calibration non supervisee : distribution des pixels et des scores des tuiles
  saines de validation (moyenne, ecart-type, quantiles). Les seuils de decision
  sont choisis ensuite, a l'etape `calibrate`.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from crackvision import tracking
from crackvision.data.prepare import read_tiles
from crackvision.models.autoencoder import (
    AnomalyDetector,
    ConvAutoencoder,
    anomaly_maps,
    image_scores,
    reconstruction_loss,
)
from crackvision.paths import Paths
from crackvision.utils import ensure_dir, get_logger, select_device, set_seed, write_json

logger = get_logger(__name__)

PIXEL_QUANTILES = (0.99, 0.995, 0.999)


class HealthyTiles(Dataset):
    def __init__(self, files: list[Path], size: int, augment: bool):
        self.files, self.size, self.augment = files, size, augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.files[index]), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                x = x[:, ::-1]
            if rng.random() < 0.5:
                x = x[::-1]
            x = np.rot90(x, int(rng.integers(0, 4)))
            x = np.clip(x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def _loader(files: list[Path], cfg: dict, augment: bool, shuffle: bool) -> DataLoader:
    return DataLoader(HealthyTiles(files, int(cfg["image_size"]), augment),
                      batch_size=int(cfg["batch_size"]), shuffle=shuffle,
                      num_workers=int(cfg["num_workers"]), drop_last=False,
                      persistent_workers=int(cfg["num_workers"]) > 0)


@torch.no_grad()
def validation_auroc(model: torch.nn.Module, loader: DataLoader, labels: np.ndarray, cfg: dict,
                     device: str) -> float:
    """AUROC du score d'anomalie sur toute la validation (tuiles saines ET fissurees)."""
    model.eval()
    scores = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        maps = anomaly_maps(x, model(x), float(cfg["ssim_weight"]), float(cfg["blur_sigma"]))
        scores.append(image_scores(maps, float(cfg["score_topk"])).cpu().numpy())
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.concatenate(scores)))


def calibrate_on_healthy(detector: AnomalyDetector, files: list[Path], cfg: dict,
                         seed: int) -> dict:
    """Statistiques de normalite : pixels et scores des tuiles saines de validation."""
    rng = np.random.default_rng(seed)
    pixels, scores = [], []
    for start in range(0, len(files), 64):
        batch = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files[start:start + 64]]
        maps, batch_scores = detector.predict(batch)
        scores.extend(batch_scores.tolist())
        for amap in maps:
            flat = amap.ravel()
            pixels.append(rng.choice(flat, size=min(flat.size, 20000), replace=False))
    pixels_np, scores_np = np.concatenate(pixels), np.asarray(scores)
    quantiles = {str(q): float(np.quantile(pixels_np, q)) for q in PIXEL_QUANTILES}
    return {
        "n_tiles": len(files),
        "pixel_mean": float(pixels_np.mean()),
        "pixel_std": float(pixels_np.std()),
        "pixel_quantiles": quantiles,
        "pixel_threshold": float(np.quantile(pixels_np, float(cfg["pixel_quantile"]))),
        "pixel_quantile": float(cfg["pixel_quantile"]),
        "score_mean": float(scores_np.mean()),
        "score_std": float(scores_np.std()),
        "score_quantiles": {str(q): float(np.quantile(scores_np, q)) for q in (0.95, 0.99)},
    }


def train_autoencoder(params: dict, paths: Paths) -> dict:
    cfg = params["autoencoder"]
    seed = int(params["seed"])
    set_seed(seed)
    device = select_device()
    model_dir = paths.model_dir("autoencoder")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)

    train_names = pd.read_csv(paths.train_balanced / "ae_train.csv")["file_name"]
    train_files = [paths.split_dir("train") / "images" / name for name in train_names]
    val_tiles = read_tiles(paths, "val")
    val_files = [paths.split_dir("val") / "images" / name
                 for name in val_tiles.loc[val_tiles["label"] == "healthy", "file_name"]]
    train_loader = _loader(train_files, cfg, augment=True, shuffle=True)
    val_loader = _loader(val_files, cfg, augment=False, shuffle=False)
    # Selection du point de controle : AUROC sur la validation annotee (par defaut).
    # Mieux reconstruire le beton sain finit par apprendre aussi a reconstruire les
    # fissures : la perte de reconstruction seule choisit un modele moins discriminant.
    selection = cfg.get("selection", "auroc")
    all_val = paths.split_dir("val") / "images"
    auroc_loader = _loader([all_val / name for name in val_tiles["file_name"]], cfg,
                           augment=False, shuffle=False)
    val_labels = (val_tiles["label"] == "positive").astype(int).to_numpy()
    eval_interval = int(cfg.get("eval_interval", 1))

    model = ConvAutoencoder(base_channels=int(cfg["base_channels"]), depth=int(cfg["depth"]),
                            latent_channels=int(cfg["latent_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                 weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    weight, noise = float(cfg["ssim_weight"]), float(cfg["noise_std"])
    logger.info("Autoencodeur : %d tuiles saines d'entrainement, %d de validation, %s",
                len(train_files), len(val_files), device)

    history, best_value, best_epoch, stale = [], -float("inf"), -1, 0
    best_loss, best_auroc = float("nan"), float("nan")
    started = time.time()
    with tracking.start_run("train_autoencoder", {"autoencoder": cfg, "seed": seed},
                            model_dir) as run:
        for epoch in range(int(cfg["epochs"])):
            model.train()
            epoch_start, train_loss, seen = time.time(), 0.0, 0
            for x in train_loader:
                x = x.to(device, non_blocking=True)
                noisy = (x + noise * torch.randn_like(x)).clamp(0, 1) if noise > 0 else x
                loss = reconstruction_loss(x, model(noisy), weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += float(loss) * len(x)
                seen += len(x)
            scheduler.step()

            model.eval()
            val_loss, val_seen = 0.0, 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device, non_blocking=True)
                    val_loss += float(reconstruction_loss(x, model(x), weight)) * len(x)
                    val_seen += len(x)
            row = {"epoch": epoch + 1, "train_loss": round(train_loss / max(seen, 1), 6),
                   "val_loss": round(val_loss / max(val_seen, 1), 6),
                   "lr": optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - epoch_start, 1)}
            last = epoch + 1 == int(cfg["epochs"])
            if selection == "auroc" and ((epoch + 1) % eval_interval == 0 or last):
                row["val_auroc"] = round(validation_auroc(model, auroc_loader, val_labels, cfg,
                                                          device), 5)
            history.append(row)
            run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            value = row.get("val_auroc") if selection == "auroc" else -row["val_loss"]
            if value is not None:
                if value > best_value:
                    best_value, best_epoch, stale = value, epoch + 1, 0
                    best_loss, best_auroc = row["val_loss"], row.get("val_auroc", float("nan"))
                    torch.save(model.state_dict(), model_dir / "model.pth")
                else:
                    stale += 1
            logger.info("Epoque %d/%d : perte %.5f, validation saine %.5f%s", epoch + 1,
                        int(cfg["epochs"]), row["train_loss"], row["val_loss"],
                        f", AUROC val {row['val_auroc']:.4f}" if "val_auroc" in row else "")
            pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)
            if int(cfg["patience"]) > 0 and stale >= int(cfg["patience"]):
                logger.info("Arret anticipe a l'epoque %d", epoch + 1)
                break
        if best_epoch < 0:  # critere jamais calculable (validation d'une seule classe)
            best_epoch, best_loss = len(history), history[-1]["val_loss"]
            torch.save(model.state_dict(), model_dir / "model.pth")

    meta = {
        "model": "autoencoder",
        "base_channels": int(cfg["base_channels"]),
        "depth": int(cfg["depth"]),
        "latent_channels": int(cfg["latent_channels"]),
        "image_size": int(cfg["image_size"]),
        "ssim_weight": weight,
        "blur_sigma": float(cfg["blur_sigma"]),
        "score_topk": float(cfg["score_topk"]),
        "noise_std": noise,
        "selection": selection,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_auroc": best_auroc,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - started, 1),
        "n_train_tiles": len(train_files),
        "n_val_healthy_tiles": len(val_files),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "torch": torch.__version__,
    }
    write_json(model_dir / "meta.json", meta)
    detector = AnomalyDetector.from_dir(model_dir, device)
    calibration = calibrate_on_healthy(detector, val_files, cfg, seed)
    write_json(model_dir / "calibration.json", calibration)
    logger.info("Calibration saine : seuil pixel %.4f, score moyen %.4f",
                calibration["pixel_threshold"], calibration["score_mean"])
    return meta
