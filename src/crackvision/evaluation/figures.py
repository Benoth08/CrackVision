"""Figures des rapports (et du README).

Conventions graphiques :
- chaque modele garde la meme couleur partout (YOLOX bleu, autoencodeur orange,
  hybride aqua) ; ce sont les trois premiers emplacements d'une palette validee
  pour les daltonismes courants ;
- la carte d'anomalie est une rampe d'une seule teinte, superposee a la photo ;
- jamais deux echelles sur un meme axe : des mesures differentes ont chacune
  leur graphique ;
- verite terrain en blanc pointille, detections en trait plein.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

from crackvision.utils import ensure_dir  # noqa: E402

MODEL_COLORS = {"yolox": "#2a78d6", "autoencoder": "#eb6834", "hybrid": "#1baf7a"}
# Libelles affiches (avec accents : ils apparaissent dans le README)
MODEL_LABELS = {"yolox": "YOLOX (supervisé)", "autoencoder": "Autoencodeur (non supervisé)",
                "hybrid": "Hybride"}
LABEL_COLORS = {"positive": "#eb6834", "healthy": "#2a78d6", "ambiguous": "#898781"}
LABEL_NAMES = {"positive": "fissurée", "healthy": "saine", "ambiguous": "ambiguë (exclue)"}
LEVEL_NAMES = {"fissure": "fissure", "anomalie_non_classee": "anomalie non classée",
               "sain": "sain"}
UNKNOWN_COLOR = "#6250d6"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    })


def _suptitle(fig, title: str) -> None:
    """Titre general au-dessus des sous-graphiques, sans chevauchement."""
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))


def _save(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_bgr: np.ndarray, amap: np.ndarray, low: float, high: float,
                    color: str = "#eb6834") -> np.ndarray:
    """Superpose la carte d'anomalie : opacite proportionnelle a l'anomalie."""
    rgb = _rgb(image_bgr).astype(np.float32) / 255.0
    alpha = np.clip((amap - low) / max(high - low, 1e-9), 0, 1)[..., None] * 0.75
    tint = np.array(matplotlib.colors.to_rgb(color), dtype=np.float32)
    return np.clip(rgb * (1 - alpha) + tint * alpha, 0, 1)


def _draw_boxes(ax, boxes, color: str, style: str = "-", width: float = 1.6,
                scores: bool = False) -> None:
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=width, linestyle=style))
        if scores and len(box) > 4:
            ax.text(x1 + 2, y1 + 2, f"{box[4]:.2f}", color="white", fontsize=7, va="top",
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})


def _image_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, fontweight="normal", loc="left", color=INK_2)
    ax.axis("off")


# --------------------------------------------------------------------------- courbes
def plot_curves(curves: dict, kind: str, path: Path, title: str, summary_key: str) -> Path:
    setup()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for model, data in curves.items():
        if kind == "pr":
            x, y = data["pr"]["recall"], data["pr"]["precision"]
        else:
            x, y = data["roc"]["fpr"], data["roc"]["tpr"]
        ax.plot(x, y, color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} ({summary_key} {data['summary']:.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        ax.set_xlabel("Taux de fausses alarmes (tuiles saines)")
        ax.set_ylabel("Rappel (tuiles fissurées)")
    else:
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left" if kind == "pr" else "lower right", fontsize=8)
    return _save(fig, path)


def plot_detection_pr(curves: dict, path: Path, title: str) -> Path:
    setup()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for model, data in curves.items():
        if data["recall"]:
            ax.plot(data["recall"], data["precision"], color=MODEL_COLORS[model],
                    label=f"{MODEL_LABELS[model]} (AP50 {data['ap50']:.3f})")
    ax.set_xlabel("Rappel (boîtes, IoU ≥ 0,5)")
    ax.set_ylabel("Précision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_metric_bars(summary: dict, metric_names: dict[str, str], path: Path,
                     title: str) -> Path:
    """Barres groupees (une couleur par modele) avec intervalle de confiance a 95 %."""
    setup()
    models = list(summary)
    metrics = list(metric_names)
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(metrics) * len(models) + 1.4))
    height = 0.8 / len(models)
    for m_index, model in enumerate(models):
        for k_index, metric in enumerate(metrics):
            value = summary[model][metric]["value"]
            low, high = summary[model][metric]["low"], summary[model][metric]["high"]
            y = k_index + (m_index - (len(models) - 1) / 2) * height
            ax.barh(y, value, height=height - 0.04, color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model] if k_index == 0 else None)
            if np.isfinite(low) and np.isfinite(high):
                ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=INK_2,
                            capsize=2, linewidth=1)
            ax.text(max(high if np.isfinite(high) else value, value) + 0.015, y,
                    f"{value:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(metrics)), [metric_names[m] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


def plot_confusions(confusions: dict, path: Path, title: str) -> Path:
    setup()
    fig, axes = plt.subplots(1, len(confusions), figsize=(3.2 * len(confusions), 3.1))
    for ax, (model, conf) in zip(np.atleast_1d(axes), confusions.items(), strict=True):
        matrix = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for (i, j), value in np.ndenumerate(matrix):
            dark = value > matrix.max() * 0.55
            ax.text(j, i, str(value), ha="center", va="center", fontsize=11,
                    color="white" if dark else INK)
        ax.set_xticks([0, 1], ["prédit sain", "prédit fissure"])
        ax.set_yticks([0, 1], ["sain", "fissure"])
        ax.grid(False)
        ax.set_title(MODEL_LABELS[model], fontsize=9)
    _suptitle(fig, title)
    return _save(fig, path)


def plot_training(yolox_history, ae_history, path: Path) -> Path:
    """Petits graphiques : perte YOLOX, AP de validation, pertes et AUROC de l'autoencodeur."""
    setup()
    with_auroc = "val_auroc" in ae_history and ae_history["val_auroc"].notna().any()
    fig, axes = plt.subplots(1, 4 if with_auroc else 3, figsize=(17 if with_auroc else 13, 3.6))
    axes[0].plot(yolox_history["epoch"], yolox_history["total_loss"], color=MODEL_COLORS["yolox"])
    axes[0].set_title("YOLOX : perte d'entraînement")
    axes[0].set_xlabel("Époque")
    evaluated = yolox_history.dropna(subset=["val_ap50"]) if "val_ap50" in yolox_history else None
    if evaluated is not None and len(evaluated):
        axes[1].plot(evaluated["epoch"], evaluated["val_ap50"], color=MODEL_COLORS["yolox"],
                     marker="o", markersize=4, label="AP50")
        axes[1].plot(evaluated["epoch"], evaluated["val_ap50_95"], color=MODEL_COLORS["yolox"],
                     linestyle="--", label="AP50:95")
        axes[1].legend(fontsize=8)
    axes[1].set_title("YOLOX : AP sur la validation")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylim(0, 1)
    axes[2].plot(ae_history["epoch"], ae_history["train_loss"], color=MODEL_COLORS["autoencoder"],
                 label="entraînement (tuiles saines)")
    axes[2].plot(ae_history["epoch"], ae_history["val_loss"], color=MODEL_COLORS["autoencoder"],
                 linestyle="--", label="validation (tuiles saines)")
    axes[2].set_title("Autoencodeur : perte de reconstruction")
    axes[2].set_xlabel("Époque")
    axes[2].legend(fontsize=8)
    if with_auroc:
        evaluated = ae_history.dropna(subset=["val_auroc"])
        axes[3].plot(evaluated["epoch"], evaluated["val_auroc"], color=MODEL_COLORS["autoencoder"],
                     marker="o", markersize=4)
        axes[3].set_title("Autoencodeur : AUROC sur la validation")
        axes[3].set_xlabel("Époque")
        axes[3].set_ylim(0.5, 1)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- exemples
def plot_examples(examples: list[dict], points: dict, heat_range: tuple[float, float],
                  path: Path, title: str) -> Path:
    """Une ligne par tuile : verite terrain, YOLOX, autoencodeur, hybride."""
    setup()
    rows = len(examples)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.25 * rows))
    axes = np.atleast_2d(axes)
    for row, example in zip(axes, examples, strict=True):
        image, mask = example["image"], example["mask"]
        truth = "fissurée" if example["label"] else "saine"
        row[0].imshow(_rgb(image))
        if mask is not None and mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                row[0].plot(contour[:, 0, 0], contour[:, 0, 1], color="white", linewidth=0.8)
        _draw_boxes(row[0], example["gts"], "white", style="--", width=1.2)
        _image_axis(row[0], f"{example['tile_id']} - vérité : {truth}")

        row[1].imshow(_rgb(image))
        kept = [d for d in example["yolox"] if d[4] >= points["yolox"]["box_threshold"]]
        _draw_boxes(row[1], kept, MODEL_COLORS["yolox"], scores=True)
        _image_axis(row[1], f"YOLOX : score max {example['yolox_score']:.2f}")

        row[2].imshow(overlay_heatmap(image, example["amap"], *heat_range))
        _image_axis(row[2], f"Autoencodeur : score {example['ae_score']:.3f}")

        row[3].imshow(_rgb(image))
        hybrid = example["hybrid"]
        _draw_boxes(row[3], hybrid["confirmed_boxes"], MODEL_COLORS["hybrid"], scores=True)
        _draw_boxes(row[3], [r["box"] for r in hybrid["unknown_regions"]], UNKNOWN_COLOR,
                    style=":", width=1.6)
        _image_axis(row[3], f"Hybride : P(défaut) {hybrid['proba']:.2f} - "
                            f"{LEVEL_NAMES[hybrid['level']]}")
    _suptitle(fig, title)
    return _save(fig, path)


def plot_full_photo(image_bgr: np.ndarray, result: dict, heat_range: tuple[float, float],
                    path: Path, title: str) -> Path:
    setup()
    height, width = image_bgr.shape[:2]
    work_w, work_h = result["work_size"]
    small = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    scale = result["scale"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 14 * height / width / 2 + 0.6))
    axes[0].imshow(_rgb(small))
    _draw_boxes(axes[0], [b[:4] * scale for b in result["yolox"]
                          if b[4] >= result["yolox_box_threshold"]],
                MODEL_COLORS["yolox"])
    _image_axis(axes[0], "YOLOX seul")
    axes[1].imshow(overlay_heatmap(small, result["anomaly_map"], *heat_range))
    _draw_boxes(axes[1], [np.r_[b[:4] * scale, b[4]] for b in result["hybrid"]["boxes"]],
                MODEL_COLORS["hybrid"], scores=True)
    _draw_boxes(axes[1], [np.array(r["box"]) * scale for r in result["hybrid"]["unknown_regions"]],
                UNKNOWN_COLOR, style=":")
    _image_axis(axes[1], "Hybride + carte d'anomalie - "
                         f"{LEVEL_NAMES[result['hybrid']['level']]}")
    _suptitle(fig, title)
    return _save(fig, path)


def plot_stress_examples(examples: list[dict], names: dict, points: dict,
                         heat_range: tuple[float, float], path: Path, title: str) -> Path:
    """Defauts hors classe incrustes : ce que voit chaque modele."""
    setup()
    fig, axes = plt.subplots(len(examples), 4, figsize=(13, 3.25 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, example in zip(axes, examples, strict=True):
        image, mask, pred = example["image"], example["mask"], example["prediction"]
        row[0].imshow(_rgb(image))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            row[0].plot(contour[:, 0, 0], contour[:, 0, 1], color="white", linewidth=0.9,
                        linestyle="--")
        _image_axis(row[0], f"Défaut incrusté : {names[example['kind']]}")
        row[1].imshow(_rgb(image))
        kept = [d for d in pred.yolox if d[4] >= points["yolox"]["box_threshold"]]
        _draw_boxes(row[1], kept, MODEL_COLORS["yolox"], scores=True)
        best = float(pred.yolox[:, 4].max()) if len(pred.yolox) else 0.0
        _image_axis(row[1], f"YOLOX : score max {best:.2f}")
        row[2].imshow(overlay_heatmap(image, pred.amap, *heat_range))
        _image_axis(row[2], f"Autoencodeur : score {pred.ae_score:.3f}")
        row[3].imshow(_rgb(image))
        _draw_boxes(row[3], pred.hybrid["confirmed_boxes"], MODEL_COLORS["hybrid"], scores=True)
        _draw_boxes(row[3], [r["box"] for r in pred.hybrid["unknown_regions"]], UNKNOWN_COLOR,
                    style=":", width=1.8)
        _image_axis(row[3], f"Hybride : P(défaut) {pred.hybrid['proba']:.2f} - "
                            f"{LEVEL_NAMES[pred.hybrid['level']]}")
    _suptitle(fig, title)
    return _save(fig, path)


def plot_stress_rates(results: dict, names: dict, path: Path) -> Path:
    """Part des tuiles signalees, par defaut hors classe et par modele."""
    setup()
    series = {"yolox": MODEL_LABELS["yolox"], "autoencoder": MODEL_LABELS["autoencoder"],
              "hybrid_avec_filet": "Hybride (décision ou filet de sécurité)"}
    colors = {"yolox": MODEL_COLORS["yolox"], "autoencoder": MODEL_COLORS["autoencoder"],
              "hybrid_avec_filet": MODEL_COLORS["hybrid"]}
    groups = ["fausses_alarmes_sans_defaut", *results["defauts"]]
    labels = ["aucun défaut\n(fausses alarmes)", *[names[k] for k in results["defauts"]]]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(series)
    for index, (key, label) in enumerate(series.items()):
        values = [results["fausses_alarmes_sans_defaut"][key]] + [
            results["defauts"][k][key] for k in results["defauts"]]
        x = np.arange(len(groups)) + (index - (len(series) - 1) / 2) * width
        ax.bar(x, values, width=width - 0.03, color=colors[key], label=label)
        for xi, value in zip(x, values, strict=True):
            ax.text(xi, value + 0.015, f"{value:.0%}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(np.arange(len(groups)), labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Part des tuiles signalées")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Défauts hors classe incrustés dans {results['n_tiles']} tuiles saines "
                 "du blind test (synthétiques)")
    handles, names_ = ax.get_legend_handles_labels()
    fig.legend(handles, names_, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, path)


def plot_gallery(items: list[dict], heat_range: tuple[float, float], path: Path, title: str) -> Path:
    """Une ligne par photo : verite terrain (masque) et prediction hybride (boites + carte)."""
    setup()
    first = items[0]["image"]
    ratio = first.shape[0] / first.shape[1]
    fig, axes = plt.subplots(len(items), 2, figsize=(15, len(items) * 7.5 * ratio + 0.8))
    axes = np.atleast_2d(axes)
    for row, item in zip(axes, items, strict=True):
        result = item["result"]
        work_w, work_h = result["work_size"]
        scale = result["scale"]
        small = cv2.resize(item["image"], (work_w, work_h), interpolation=cv2.INTER_AREA)
        mask = item["mask"]
        if mask.shape != (work_h, work_w):
            mask = cv2.resize(mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

        truth = _rgb(small).copy()
        truth[mask > 0] = (255, 255, 255)
        row[0].imshow(truth)
        _image_axis(row[0], f"Photo {item['photo_id']} ({item['reason']}) - masque de vérité en blanc")

        row[1].imshow(overlay_heatmap(small, result["anomaly_map"], *heat_range))
        _draw_boxes(row[1], [np.r_[b[:4] * scale, b[4]] for b in result["hybrid"]["boxes"]],
                    MODEL_COLORS["hybrid"], scores=True, width=2)
        _draw_boxes(row[1], [np.array(r["box"]) * scale for r in result["hybrid"]["unknown_regions"]],
                    UNKNOWN_COLOR, style=":", width=2)
        errors = f"{item['tiles_fn']} tuile(s) fissurée(s) manquée(s), {item['tiles_fp']} fausse(s) alarme(s)"
        _image_axis(row[1], f"Hybride : {item['n_boxes']} fissure(s), {item['n_unknown']} anomalie(s) "
                            f"non classée(s) - {errors}")
    _suptitle(fig, title)
    return _save(fig, path)


# --------------------------------------------------------------------------- donnees
def plot_data_samples(samples: list[dict], path: Path, title: str) -> Path:
    setup()
    cols = min(4, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, sample in zip(axes, samples, strict=False):
        ax.imshow(_rgb(sample["image"]))
        _draw_boxes(ax, sample["gts"], "white", style="--", width=1.2)
        _image_axis(ax, f"{sample['tile_id']} ({LABEL_NAMES[sample['label']]})")
    for ax in axes[len(samples):]:
        ax.axis("off")
    _suptitle(fig, title)
    return _save(fig, path)


def plot_split_distribution(counts: dict, path: Path) -> Path:
    """Tuiles par jeu et par etiquette (barres empilees, ecart de 2 px entre segments)."""
    setup()
    splits = list(counts)
    labels = ["positive", "healthy", "ambiguous"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(splits))
    for label in labels:
        values = np.array([counts[s].get(label, 0) for s in splits], dtype=float)
        ax.barh(splits, values, left=left, color=LABEL_COLORS[label], label=LABEL_NAMES[label],
                edgecolor=SURFACE, linewidth=2, height=0.6)
        for index, value in enumerate(values):
            if value > 0.06 * max(left.max() + values.max(), 1):
                ax.text(left[index] + value / 2, index, f"{int(value)}", ha="center",
                        va="center", fontsize=8, color="white")
        left += values
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Nombre de tuiles")
    ax.set_title("Tuiles par jeu (découpage stratifié par photo)")
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def plot_balance(report: dict, path: Path) -> Path:
    """Distribution naturelle et reequilibree du jeu d'entrainement."""
    setup()
    strata = report["natural"]["strata"]
    fig, axes = plt.subplots(1, len(strata) + 1, figsize=(4 * (len(strata) + 1), 3.4))
    natural, balanced = "#c3c2b7", "#2a78d6"
    pos = [report["natural"]["positive_fraction"], report["balanced"]["positive_fraction"]]
    axes[0].bar(["naturel", "rééquilibré"], pos, color=[natural, balanced], width=0.55)
    for index, value in enumerate(pos):
        axes[0].text(index, value + 0.02, f"{value:.0%}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Part de tuiles fissurées")
    titles = {"size_bin": "Taille de fissure", "orientation": "Orientation",
              "multiplicity": "Nombre de fissures"}
    for ax, column in zip(axes[1:], strata, strict=True):
        keys = list(strata[column])
        x = np.arange(len(keys))
        before = [strata[column][k] for k in keys]
        after = [report["balanced"]["strata"][column].get(k, 0) for k in keys]
        ax.bar(x - 0.2, before, width=0.38, color=natural, label="naturel")
        ax.bar(x + 0.2, after, width=0.38, color=balanced, label="rééquilibré")
        ax.set_xticks(x, keys)
        ax.set_ylim(0, 1)
        ax.set_title(titles.get(column, column))
        ax.grid(axis="x", visible=False)
    axes[-1].legend(fontsize=8)
    _suptitle(fig, "Jeu d'entraînement YOLOX : avant et après rééquilibrage")
    return _save(fig, path)
