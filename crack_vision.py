# -*- coding: utf-8 -*-
"""
Detection et Metrologie de Fissures dans le Beton
Modele : YOLO11n (Ultralytics)
Dataset : Concrete Crack Detection (Roboflow Universe)
"""

import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")

print("[OK] Imports de base termines.")

from ultralytics import YOLO
import ultralytics

print(f"[OK] Ultralytics version : {ultralytics.__version__}")
print("[OK] YOLO11 disponible.")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"     GPU : {torch.cuda.get_device_name(0)}")

# Option A : Via l'API Roboflow (recommande)
# Vous devez creer un compte gratuit sur https://roboflow.com
# et obtenir votre cle API dans les parametres du compte.
#
# Le dataset utilise ici est "Concrete Crack Detection" disponible sur
# Roboflow Universe. Plusieurs datasets sont disponibles ; nous utilisons
# celui avec le plus d'images et des annotations bounding box.

from roboflow import Roboflow

# --- CONFIGURATION (A MODIFIER) ---
# Remplacez par votre cle API Roboflow
ROBOFLOW_API_KEY = "VOTRE_CLE_API"  # Obtenez-la sur https://app.roboflow.com/settings/api-key

# --- TELECHARGEMENT ---
print("Telechargement du dataset depuis Roboflow Universe...")
print("Si vous n'avez pas de cle API, suivez l'Option B ci-dessous.\n")

DATA_DIR = "data"
DATASET_YAML = None

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    # Dataset "Concrete Crack" par vijayalakshmi (~1300 images, 1 classe)
    project = rf.workspace("vijayalakshmi-2yshx").project(
        "concrete-surface-crack-detection-using-yolov5-model"
    )
    version = project.version(3)
    dataset = version.download("yolov11", location=DATA_DIR)
    DATASET_YAML = os.path.join(DATA_DIR, "data.yaml")
    print(f"[OK] Dataset telecharge dans : {DATA_DIR}")
    print(f"[OK] Config YAML : {DATASET_YAML}")
except Exception as e:
    print(f"[INFO] Telechargement automatique non disponible : {e}")
    print("\n--- OPTION B : Telechargement manuel ---")
    print("1. Allez sur https://universe.roboflow.com")
    print("2. Cherchez 'concrete crack detection'")
    print("3. Choisissez un dataset avec bounding boxes")
    print("4. Cliquez 'Download Dataset' > Format 'YOLOv11'")
    print("5. Dezippez dans le dossier 'data/'")
    print("6. Mettez a jour DATASET_YAML ci-dessous\n")

# Si telechargement manuel, specifiez le chemin ici :
if DATASET_YAML is None:
    DATASET_YAML = "data/data.yaml"

    """Explore la structure du dataset et affiche des statistiques."""
    data_path = Path(data_dir)

    # Compter les images et labels
    stats = {}
    for split in ["train", "valid", "test"]:
        img_dir = data_path / split / "images"
        lbl_dir = data_path / split / "labels"
        if img_dir.exists():
            n_images = len(list(img_dir.glob("*.*")))
            n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            stats[split] = {"images": n_images, "labels": n_labels}

    print("--- Structure du dataset ---")
    for split, info in stats.items():
        print(f"  {split:6s} : {info['images']:4d} images, {info['labels']:4d} labels")

    # Analyser les annotations (tailles des bounding boxes)
    all_widths = []
    all_heights = []
    all_aspects = []

    train_labels = data_path / "train" / "labels"
    if train_labels.exists():
        for txt_file in train_labels.glob("*.txt"):
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w = float(parts[3])
                        h = float(parts[4])
                        all_widths.append(w)
                        all_heights.append(h)
                        if h > 0:
                            all_aspects.append(w / h)

    if all_widths:
        print(f"\n--- Statistiques des bounding boxes (train) ---")
        print(f"  Nombre total de fissures annotees : {len(all_widths)}")
        print(f"  Largeur moyenne (norm) : {np.mean(all_widths):.4f}")
        print(f"  Hauteur moyenne (norm) : {np.mean(all_heights):.4f}")
        print(f"  Ratio d'aspect moyen   : {np.mean(all_aspects):.2f}")

    return stats, all_widths, all_heights, all_aspects


stats, widths, heights, aspects = explore_dataset(DATA_DIR)

# Visualiser les distributions
if widths:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribution des largeurs (normalise)")
    axes[0].set_xlabel("Largeur relative")

    axes[1].hist(heights, bins=30, color="coral", edgecolor="white", alpha=0.8)
    axes[1].set_title("Distribution des hauteurs (normalise)")
    axes[1].set_xlabel("Hauteur relative")

    axes[2].hist(aspects, bins=30, color="green", edgecolor="white", alpha=0.8)
    axes[2].set_title("Distribution des ratios d'aspect (w/h)")
    axes[2].set_xlabel("Ratio w/h")
    axes[2].axvline(1.0, color="red", linestyle="--", alpha=0.7, label="Carre")
    axes[2].legend()

    plt.suptitle("Analyse des bounding boxes du dataset", fontsize=13)
    plt.tight_layout()
    plt.show()

# Afficher des exemples d'images annotees
def show_annotated_samples(data_dir: str, split: str = "train", n_samples: int = 6):
    """Affiche des images avec leurs bounding boxes."""
    img_dir = Path(data_dir) / split / "images"
    lbl_dir = Path(data_dir) / split / "labels"

    if not img_dir.exists():
        print(f"[INFO] Dossier {img_dir} non trouve.")
        return

    img_files = sorted(list(img_dir.glob("*.*")))[:n_samples]
    n = min(len(img_files), n_samples)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        axes[i].imshow(img_rgb)

        # Charger les annotations
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w_img
                        yc = float(parts[2]) * h_img
                        bw = float(parts[3]) * w_img
                        bh = float(parts[4]) * h_img
                        x1 = xc - bw / 2
                        y1 = yc - bh / 2
                        rect = patches.Rectangle(
                            (x1, y1), bw, bh,
                            linewidth=2, edgecolor="lime", facecolor="none"
                        )
                        axes[i].add_patch(rect)

        axes[i].set_title(img_path.name, fontsize=8)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Exemples du dataset avec annotations", fontsize=13)
    plt.tight_layout()
    plt.show()


show_annotated_samples(DATA_DIR)

print("FINE-TUNING YOLO11n sur le dataset de fissures")
print("=" * 60)

# Charger le modele pre-entraine
model = YOLO("yolo11n.pt")

# Lancer l'entrainement
NUM_EPOCHS = 50  # Ajuster selon le temps disponible (50 = bon compromis)

results = model.train(
    data=DATASET_YAML,
    epochs=NUM_EPOCHS,
    imgsz=640,
    batch=16 if DEVICE == "cuda" else 8,
    device=DEVICE,
    patience=10,           # Early stopping si pas d'amelioration pendant 10 epochs
    save=True,
    project="runs/detect",
    name="crack_detection",
    exist_ok=True,         # Ecraser si le dossier existe deja
    pretrained=True,
    verbose=True
)

print("\n[OK] Entrainement termine.")

# Chemin vers le meilleur modele
BEST_MODEL_PATH = "runs/detect/crack_detection/weights/best.pt"
print(f"Meilleur modele : {BEST_MODEL_PATH}")

results_dir = Path("runs/detect/crack_detection")

# Afficher les courbes d'entrainement si disponibles
results_img = results_dir / "results.png"
if results_img.exists():
    fig, ax = plt.subplots(figsize=(14, 8))
    img = plt.imread(str(results_img))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Courbes d'entrainement YOLO11", fontsize=14)
    plt.tight_layout()
    plt.show()

# Afficher la matrice de confusion
cm_img = results_dir / "confusion_matrix.png"
if cm_img.exists():
    fig, ax = plt.subplots(figsize=(8, 6))
    img = plt.imread(str(cm_img))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Matrice de confusion", fontsize=14)
    plt.tight_layout()
    plt.show()

# Afficher la courbe F1
f1_img = results_dir / "F1_curve.png"
if f1_img.exists():
    fig, ax = plt.subplots(figsize=(8, 5))
    img = plt.imread(str(f1_img))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Courbe F1 vs seuil de confiance", fontsize=14)
    plt.tight_layout()
    plt.show()

print("INFERENCE + METROLOGIE DES FISSURES")
print("=" * 60)

# Charger le meilleur modele
if Path(BEST_MODEL_PATH).exists():
    best_model = YOLO(BEST_MODEL_PATH)
    print(f"[OK] Modele charge : {BEST_MODEL_PATH}")
else:
    print("[INFO] Modele fine-tune non trouve, utilisation du modele pre-entraine.")
    best_model = YOLO("yolo11n.pt")


def run_inference_with_metrology(
    model,
    image_path: str,
    confidence: float = 0.25,
    pixel_per_mm: float = None
) -> tuple:
    """Execute l'inference YOLO11 et extrait les mesures metrologiques.

    Args:
        model: Modele YOLO charge.
        image_path: Chemin vers l'image.
        confidence: Seuil de confiance minimum.
        pixel_per_mm: Facteur de calibration (None = mesures en pixels uniquement).

    Returns:
        (image_annotee, dataframe_mesures, resultats_bruts)
    """
    # Inference
    results = model.predict(
        image_path,
        conf=confidence,
        verbose=False
    )
    result = results[0]

    # Charger l'image originale pour les dimensions
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # Extraire les detections
    boxes = result.boxes
    n_detections = len(boxes)

    measurements = []

    if n_detections > 0:
        xyxy = boxes.xyxy.cpu().numpy()   # (n, 4) : x1, y1, x2, y2 en pixels
        xywh = boxes.xywh.cpu().numpy()   # (n, 4) : x_center, y_center, w, h en pixels
        confs = boxes.conf.cpu().numpy()   # (n,)
        classes = boxes.cls.cpu().numpy()  # (n,)

        for i in range(n_detections):
            x1, y1, x2, y2 = xyxy[i]
            xc, yc, w_px, h_px = xywh[i]
            conf = confs[i]

            # Surface de la bounding box en pixels
            surface_px = w_px * h_px

            # Ratio d'aspect (w/h) : les fissures sont typiquement allongees
            aspect_ratio = w_px / h_px if h_px > 0 else 0

            # Diagonale de la bounding box (estimation grossiere de la longueur)
            diagonal_px = np.sqrt(w_px**2 + h_px**2)

            measurement = {
                "id": i + 1,
                "confiance": round(float(conf), 3),
                "x_centre_px": round(float(xc), 1),
                "y_centre_px": round(float(yc), 1),
                "largeur_px": round(float(w_px), 1),
                "hauteur_px": round(float(h_px), 1),
                "surface_px2": round(float(surface_px), 1),
                "diagonale_px": round(float(diagonal_px), 1),
                "ratio_aspect": round(float(aspect_ratio), 2),
            }

            # Conversion en mm si facteur de calibration fourni
            if pixel_per_mm is not None and pixel_per_mm > 0:
                measurement["largeur_mm"] = round(float(w_px / pixel_per_mm), 2)
                measurement["hauteur_mm"] = round(float(h_px / pixel_per_mm), 2)
                measurement["surface_mm2"] = round(float(surface_px / pixel_per_mm**2), 2)
                measurement["diagonale_mm"] = round(float(diagonal_px / pixel_per_mm), 2)

            measurements.append(measurement)

    # Creer le DataFrame
    df = pd.DataFrame(measurements)

    # Image annotee (generee par Ultralytics)
    annotated_img = result.plot()  # Image BGR avec annotations

    return annotated_img, df, result


# --- Test sur des images du dataset ---
test_dir = Path(DATA_DIR) / "valid" / "images"
if not test_dir.exists():
    test_dir = Path(DATA_DIR) / "test" / "images"

if test_dir.exists():
    test_images = sorted(list(test_dir.glob("*.*")))[:6]
else:
    test_images = []

if test_images:
    print(f"\nInference sur {len(test_images)} images de validation...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    all_dfs = []

    for i, img_path in enumerate(test_images):
        annotated, df, _ = run_inference_with_metrology(
            best_model, str(img_path), confidence=0.25
        )

        # Afficher l'image annotee
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated_rgb)
        axes[i].set_title(f"{img_path.name}\n{len(df)} fissure(s)", fontsize=9)
        axes[i].axis("off")

        if not df.empty:
            all_dfs.append(df)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Resultats d'inference YOLO11 -- Detection de fissures", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Afficher le tableau metrologique d'un exemple
    if all_dfs:
        print("\n--- Tableau metrologique (premier exemple avec detections) ---")
        sample_df = all_dfs[0]
        display_cols = ["id", "confiance", "largeur_px", "hauteur_px",
                        "surface_px2", "diagonale_px", "ratio_aspect"]
        print(sample_df[display_cols].to_string(index=False))
else:
    print("[INFO] Pas d'images de test trouvees. Uploadez une image manuellement.")

    # Concatener toutes les mesures
    all_measurements = pd.concat(all_dfs, ignore_index=True)

    print(f"\n--- Statistiques globales sur {len(all_measurements)} fissures detectees ---")
    print(f"  Surface moyenne (px2)   : {all_measurements['surface_px2'].mean():.0f}")
    print(f"  Surface mediane (px2)   : {all_measurements['surface_px2'].median():.0f}")
    print(f"  Confiance moyenne       : {all_measurements['confiance'].mean():.3f}")
    print(f"  Ratio d'aspect moyen    : {all_measurements['ratio_aspect'].mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribution des surfaces
    axes[0].hist(all_measurements["surface_px2"], bins=20,
                 color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribution des surfaces (px2)")
    axes[0].set_xlabel("Surface bounding box (pixels carres)")

    # Confiance vs Surface
    axes[1].scatter(all_measurements["surface_px2"],
                    all_measurements["confiance"],
                    alpha=0.6, color="coral", s=30)
    axes[1].set_title("Confiance vs Surface")
    axes[1].set_xlabel("Surface (px2)")
    axes[1].set_ylabel("Confiance")

    # Distribution des ratios d'aspect
    axes[2].hist(all_measurements["ratio_aspect"], bins=20,
                 color="green", edgecolor="white", alpha=0.8)
    axes[2].axvline(1.0, color="red", linestyle="--", label="Carre (r=1)")
    axes[2].set_title("Ratios d'aspect des fissures")
    axes[2].set_xlabel("Ratio largeur/hauteur")
    axes[2].legend()

    plt.suptitle("Analyse metrologique des fissures detectees", fontsize=13)
    plt.tight_layout()
    plt.show()

print("EXEMPLE : Conversion pixels -> millimetres")
print("=" * 60)

# Simulation d'une calibration
# Hypothese : on a mesure qu'un objet de 50 mm fait 250 pixels sur l'image
CALIBRATION_OBJECT_MM = 50.0
CALIBRATION_OBJECT_PX = 250.0
PIXEL_PER_MM = CALIBRATION_OBJECT_PX / CALIBRATION_OBJECT_MM

print(f"Facteur de calibration : {PIXEL_PER_MM:.1f} pixels/mm")
print(f"  (Objet de reference : {CALIBRATION_OBJECT_MM} mm = {CALIBRATION_OBJECT_PX} px)")

if test_images:
    annotated, df_calib, _ = run_inference_with_metrology(
        best_model,
        str(test_images[0]),
        confidence=0.25,
        pixel_per_mm=PIXEL_PER_MM
    )

    if not df_calib.empty:
        print(f"\n--- Mesures avec calibration (image : {test_images[0].name}) ---")
        calib_cols = [c for c in df_calib.columns if c != "x_centre_px" and c != "y_centre_px"]
        print(df_calib[calib_cols].to_string(index=False))
    else:
        print("  Aucune fissure detectee sur cette image.")

print("EVALUATION FORMELLE (mAP, Precision, Rappel)")
print("=" * 60)

# Evaluation sur le set de validation
if Path(BEST_MODEL_PATH).exists():
    eval_model = YOLO(BEST_MODEL_PATH)
else:
    eval_model = best_model

metrics = eval_model.val(data=DATASET_YAML, verbose=False)

print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
print(f"  Precision       : {metrics.box.mp:.4f}")
print(f"  Rappel          : {metrics.box.mr:.4f}")

"""
CrackVision -- Interface Streamlit
Detection et metrologie de fissures dans le beton (YOLO11)
Lancer avec : streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# --- Configuration ---
st.set_page_config(
    page_title="CrackVision -- Detection de fissures",
    layout="wide"
)
st.title("CrackVision -- Detection de Fissures (YOLO11)")
st.caption("Detection automatique et metrologie de fissures sur surfaces beton")

# --- Sidebar ---
with st.sidebar:
    st.header("Parametres")
    confidence = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    st.divider()
    st.header("Calibration (optionnel)")
    use_calibration = st.checkbox("Activer la conversion en mm")
    pixel_per_mm = None
    if use_calibration:
        ref_mm = st.number_input("Taille objet reference (mm)", value=50.0, min_value=0.1)
        ref_px = st.number_input("Taille objet reference (pixels)", value=250.0, min_value=1.0)
        pixel_per_mm = ref_px / ref_mm
        st.info(f"Facteur : {pixel_per_mm:.1f} px/mm")

# --- Chargement du modele ---
@st.cache_resource
def load_model():
    model_path = "runs/detect/crack_detection/weights/best.pt"
    if Path(model_path).exists():
        return YOLO(model_path)
    return YOLO("yolo11n.pt")

model = load_model()

# --- Upload ---
uploaded = st.file_uploader(
    "Deposez une image de surface beton",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded:
    # Charger l'image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_img, w_img = image.shape[:2]

    # Inference
    with st.spinner("Detection en cours..."):
        results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]

    # Image annotee
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Extraire les mesures
    boxes = result.boxes
    measurements = []

    if len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            xc, yc, w_px, h_px = xywh[i]
            surface_px = w_px * h_px
            diag_px = np.sqrt(w_px**2 + h_px**2)
            aspect = w_px / h_px if h_px > 0 else 0

            m = {
                "Fissure": i + 1,
                "Confiance": f"{confs[i]:.1%}",
                "Largeur (px)": f"{w_px:.0f}",
                "Hauteur (px)": f"{h_px:.0f}",
                "Surface (px2)": f"{surface_px:.0f}",
                "Diagonale (px)": f"{diag_px:.0f}",
                "Ratio w/h": f"{aspect:.2f}",
            }

            if pixel_per_mm is not None:
                m["Largeur (mm)"] = f"{w_px / pixel_per_mm:.1f}"
                m["Hauteur (mm)"] = f"{h_px / pixel_per_mm:.1f}"
                m["Surface (mm2)"] = f"{surface_px / pixel_per_mm**2:.1f}"

            measurements.append(m)

    # Affichage
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection")
        st.image(annotated_rgb, use_container_width=True)
        st.caption(f"Image : {w_img} x {h_img} px | {len(boxes)} fissure(s) detectee(s)")

    with col2:
        st.subheader("Bilan")
        if len(boxes) > 0:
            st.error(f"{len(boxes)} fissure(s) detectee(s)")
        else:
            st.success("Aucune fissure detectee")

        st.metric("Dimensions image", f"{w_img} x {h_img} px")

        if pixel_per_mm:
            st.metric("Facteur calibration", f"{pixel_per_mm:.1f} px/mm")

    # Tableau metrologique
    if measurements:
        st.subheader("Tableau metrologique")
        df = pd.DataFrame(measurements)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            "Note : les dimensions correspondent a la bounding box (rectangle englobant), "
            "pas a la fissure elle-meme. La surface est une estimation par exces. "
            "Pour une mesure precise, une segmentation d'instance serait necessaire."
        )
    else:
        st.info("Aucune fissure detectee avec ce seuil de confiance. "
                "Essayez de baisser le seuil dans la barre laterale.")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("[OK] Fichier app.py genere.")
print("Pour lancer l'interface : streamlit run app.py")

print("RESUME -- CRACKVISION")
print("=" * 60)

print(f"""
Modele       : YOLO11n (fine-tune sur fissures beton)
Dataset      : Concrete Crack Detection (Roboflow Universe)
Entrainement : {NUM_EPOCHS} epochs, imgsz=640, device={DEVICE}

Fichiers generes :
  - runs/detect/crack_detection/weights/best.pt  (modele)
  - app.py  (interface Streamlit)

Pipeline :
  Image -> YOLO11 inference -> Bounding boxes -> Metrologie (px et mm)

Points cles pour l'entretien :
  1. YOLO11 est anchor-free avec tetes decouplees (sept 2024)
  2. La metrologie par bounding box est une estimation enveloppe
  3. La conversion px->mm necessite un facteur de calibration
  4. Les fissures diagonales sont surestimees (sqrt(2) factor)
  5. Extension naturelle : segmentation d'instance pour mesures precises
""")

print("[OK] Projet CrackVision termine.")
