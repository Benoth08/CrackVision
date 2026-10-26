"""Etape `train_yolox` : affinage de YOLOX sur les tuiles de fissures.

On reutilise les briques officielles de YOLOX (Exp, jeu COCO, augmentation
mosaique, optimiseur SGD par groupes, LR "yoloxwarmcos", EMA) mais avec une boucle
d'entrainement independante du materiel : le Trainer officiel suppose CUDA
(`torch.cuda.set_device`, evaluateur en `torch.cuda.FloatTensor`), or le
pipeline doit aussi tourner sur CPU (tests de fumee, integration continue).

Le meilleur modele est choisi sur l'AP50 de la validation ; le blind test n'est
jamais lu ici.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from crackvision import tracking
from crackvision.data.prepare import read_tiles
from crackvision.data.sources import download
from crackvision.evaluation.metrics import coco_detection_metrics, image_metrics
from crackvision.models.yolox_model import VARIANTS, YoloxDetector, build_model
from crackvision.paths import Paths
from crackvision.utils import (
    ensure_dir,
    get_logger,
    read_json,
    select_device,
    set_seed,
    sha256_file,
    write_json,
)

logger = get_logger(__name__)

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"


def make_exp(cfg: dict, paths: Paths, seed: int):
    """Experience YOLOX parametree par params.yaml (au lieu d'un fichier exps/*.py)."""
    from yolox.exp import Exp

    class CrackExp(Exp):
        def __init__(self):
            super().__init__()
            variant = VARIANTS[cfg["variant"]]
            self.num_classes = 1
            self.depth, self.width = variant.depth, variant.width
            self.variant = cfg["variant"]
            size = int(cfg["input_size"])
            self.input_size = (size, size)
            self.test_size = (size, size)
            self.multiscale_range = int(cfg["multiscale_range"])
            self.data_num_workers = int(cfg["num_workers"])
            self.mosaic_prob = float(cfg["mosaic_prob"])
            self.mixup_prob = float(cfg["mixup_prob"])
            self.enable_mixup = self.mixup_prob > 0
            self.hsv_prob = float(cfg["hsv_prob"])
            self.flip_prob = float(cfg["flip_prob"])
            self.degrees = float(cfg["degrees"])
            self.translate = float(cfg["translate"])
            self.mosaic_scale = tuple(float(v) for v in cfg["mosaic_scale"])
            self.shear = float(cfg["shear"])
            self.warmup_epochs = int(cfg["warmup_epochs"])
            self.max_epoch = int(cfg["epochs"])
            self.no_aug_epochs = int(cfg["no_aug_epochs"])
            self.min_lr_ratio = float(cfg["min_lr_ratio"])
            self.basic_lr_per_img = float(cfg["basic_lr_per_img"])
            self.weight_decay = float(cfg["weight_decay"])
            self.ema = bool(cfg["ema"])
            self.test_conf = float(cfg["test_conf"])
            self.nmsthre = float(cfg["nms"])
            self.seed = seed
            self.output_dir = str(paths.model_dir("yolox"))
            self.exp_name = f"crackvision_yolox_{self.variant}"

        def get_model(self):
            if getattr(self, "model", None) is None:
                self.model = build_model(self.variant, self.num_classes, self.act)
            self.model.train()
            return self.model

        def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
            from yolox.data import COCODataset, TrainTransform

            # Les images restent dans data/processed/train/images : le jeu equilibre
            # ne contient que des annotations (avec des tuiles repetees).
            return COCODataset(
                data_dir=str(paths.train_balanced),
                json_file="instances.json",
                name="../train/images",
                img_size=self.input_size,
                preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob,
                                       hsv_prob=self.hsv_prob),
                cache=False,
            )

    return CrackExp()


def fetch_pretrained(cfg: dict, paths: Paths) -> Path:
    weights = cfg["pretrained_weights"][cfg["variant"]]
    destination = paths.external / "weights" / f"yolox_{cfg['variant']}.pth"
    return download(weights["url"], destination, weights["sha256"])


def load_pretrained(model: torch.nn.Module, weights: Path) -> None:
    from yolox.utils import load_ckpt

    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        # Poids officiels verifies par empreinte : le chargement complet est sur.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    # Les couches de classification (80 classes COCO -> 1) sont ignorees par load_ckpt.
    load_ckpt(model, checkpoint["model"])


def random_input_size(base: int, spread: int, rng: random.Random) -> tuple[int, int]:
    if spread <= 0:
        return (base, base)
    size = 32 * rng.randint(base // 32 - spread, base // 32 + spread)
    return (size, size)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return image


def evaluate_on_val(model: torch.nn.Module, cfg: dict, paths: Paths, device: str) -> dict:
    """AP COCO et AUROC (score maximal par tuile) sur la validation."""
    gt = read_json(paths.split_dir("val") / "annotations" / "instances.json")
    image_dir = paths.split_dir("val") / "images"
    detector = YoloxDetector(model, int(cfg["input_size"]), float(cfg["nms"]), 1, device)
    predictions, max_scores, labels = {}, [], []
    for start in range(0, len(gt["images"]), 32):
        batch = gt["images"][start:start + 32]
        dets = detector.predict([read_bgr(image_dir / img["file_name"]) for img in batch],
                                conf=float(cfg["test_conf"]))
        for img, det in zip(batch, dets, strict=True):
            predictions[img["id"]] = det
            max_scores.append(float(det[:, 4].max()) if len(det) else 0.0)
            labels.append(1 if img["label"] == "positive" else 0)
    metrics = coco_detection_metrics(gt, predictions)
    metrics.pop("pr_curve")
    labels_np, scores_np = np.array(labels), np.array(max_scores)
    metrics["image_auroc"] = image_metrics(labels_np, scores_np, 0.5)["auroc"]
    return metrics


def train_yolox(params: dict, paths: Paths) -> dict:
    cfg = params["yolox"]
    seed = int(params["seed"])
    set_seed(seed)
    device = select_device()
    use_amp = bool(cfg["amp"]) and device.startswith("cuda")
    model_dir = paths.model_dir("yolox")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)

    exp = make_exp(cfg, paths, seed)
    model = exp.get_model()
    pretrained_info = None
    if cfg["pretrained"]:
        weights = fetch_pretrained(cfg, paths)
        load_pretrained(model, weights)
        pretrained_info = {"url": cfg["pretrained_weights"][cfg["variant"]]["url"],
                           "sha256": sha256_file(weights)}
    model.to(device)

    batch_size = int(cfg["batch_size"])
    optimizer = exp.get_optimizer(batch_size)
    train_loader = exp.get_data_loader(batch_size=batch_size, is_distributed=False, no_aug=False)
    max_iter = len(train_loader)
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * batch_size, max_iter)
    ema = None
    if exp.ema:
        from yolox.utils import ModelEMA

        ema = ModelEMA(model, 0.9998)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_iter = iter(train_loader)
    rng = random.Random(seed)
    input_size = exp.input_size

    logger.info("YOLOX-%s : %d images equilibrees, %d iterations/epoque, %s%s", exp.variant,
                len(exp.dataset), max_iter, device, " (AMP)" if use_amp else "")
    history, best_value, best_epoch, stale = [], -math.inf, -1, 0
    selection = cfg["selection_metric"]
    started = time.time()
    with tracking.start_run("train_yolox", {"yolox": cfg, "seed": seed}, model_dir) as run:
        for epoch in range(exp.max_epoch):
            if exp.no_aug_epochs > 0 and epoch == exp.max_epoch - exp.no_aug_epochs:
                logger.info("Dernieres epoques sans mosaique, avec perte L1")
                train_loader.close_mosaic()
                model.head.use_l1 = True
            model.train()
            epoch_start = time.time()
            sums: dict[str, float] = {}
            for iteration in range(max_iter):
                inputs, targets, _, _ = next(data_iter)
                inputs = inputs.to(device, non_blocking=True).float()
                targets = targets.to(device, non_blocking=True).float()
                targets.requires_grad = False
                inputs, targets = exp.preprocess(inputs, targets, input_size)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(inputs, targets)
                loss = outputs["total_loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                progress = epoch * max_iter + iteration
                lr = lr_scheduler.update_lr(progress + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                for key, value in outputs.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                if (progress + 1) % 10 == 0:
                    input_size = random_input_size(int(cfg["input_size"]),
                                                   exp.multiscale_range, rng)

            row = {"epoch": epoch + 1, "lr": lr, "seconds": round(time.time() - epoch_start, 1)}
            row.update({k: round(v / max_iter, 5) for k, v in sums.items()})
            last_epoch = epoch + 1 == exp.max_epoch
            if (epoch + 1) % int(cfg["eval_interval"]) == 0 or last_epoch or epoch + 1 > (
                    exp.max_epoch - exp.no_aug_epochs):
                eval_model = ema.ema if ema is not None else model
                metrics = evaluate_on_val(eval_model, cfg, paths, device)
                row.update({f"val_{k}": round(v, 5) for k, v in metrics.items()})
                value = metrics[selection]
                if value > best_value:
                    best_value, best_epoch, stale = value, epoch + 1, 0
                    torch.save(eval_model.state_dict(), model_dir / "model.pth")
                else:
                    stale += 1
                logger.info("Epoque %d/%d : perte %.3f, val AP50 %.4f, AP50:95 %.4f (meilleure %.4f)",
                            epoch + 1, exp.max_epoch, row["total_loss"], metrics["ap50"],
                            metrics["ap50_95"], best_value)
            else:
                logger.info("Epoque %d/%d : perte %.3f", epoch + 1, exp.max_epoch,
                            row["total_loss"])
            history.append(row)
            run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)
            if int(cfg["patience"]) > 0 and stale >= int(cfg["patience"]):
                logger.info("Arret anticipe : %d evaluations sans progres", stale)
                break

    best_row = next(r for r in history if r["epoch"] == best_epoch)
    meta = {
        "model": "yolox",
        "variant": exp.variant,
        "num_classes": 1,
        "class_names": ["fissure"],
        "input_size": int(cfg["input_size"]),
        "nms": float(cfg["nms"]),
        "test_conf": float(cfg["test_conf"]),
        "prepare": {"scale": float(params["prepare"]["scale"]),
                    "tile_size": int(params["prepare"]["tile_size"])},
        "best_epoch": best_epoch,
        "selection_metric": selection,
        "val_metrics": {k[4:]: v for k, v in best_row.items() if k.startswith("val_")},
        "epochs_run": len(history),
        "train_seconds": round(time.time() - started, 1),
        "device": device if not device.startswith("cuda") else torch.cuda.get_device_name(0),
        "amp": use_amp,
        "pretrained": pretrained_info,
        "yolox_commit": YOLOX_COMMIT,
        "train_samples": len(exp.dataset),
        "train_annotations_sha256": sha256_file(
            paths.train_balanced / "annotations" / "instances.json"),
        "n_val_tiles": len(read_tiles(paths, "val")),
        "torch": torch.__version__,
    }
    write_json(model_dir / "meta.json", meta)
    logger.info("Meilleur modele : epoque %d, %s = %.4f", best_epoch, selection, best_value)
    return meta
