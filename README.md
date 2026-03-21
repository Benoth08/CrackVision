# CrackVision -- Detection et Metrologie de Fissures dans le Beton

> Detection automatique de fissures par YOLO11 + estimation dimensionnelle pour l'inspection d'ouvrages d'art

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLO11](https://img.shields.io/badge/Model-YOLO11n-purple)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

L'inspection visuelle des ouvrages d'art (ponts, tunnels, barrages, centrales nucleaires) est realisee manuellement par des inspecteurs qui photographient les surfaces et mesurent les fissures au fissurometre. Ce processus coute **2 000 a 10 000 euros par jour**, reste subjectif (deux inspecteurs ne reportent pas les memes defauts) et peut etre dangereux (zones en hauteur, environnements irradies).

## La solution

Un systeme qui prend une photo de surface beton, **detecte automatiquement les fissures** via YOLO11 (etat de l'art, septembre 2024) et produit un **rapport metrologique** avec localisation et dimensions de chaque defaut. Tout tourne en local, sans cloud.

## Ce qui distingue ce projet

Ce n'est pas "juste de la detection d'objets". Le pipeline couple l'IA (YOLO11) a une **reflexion metrologique de physicien** :
- Extraction des dimensions (largeur, hauteur, surface, diagonale) en pixels
- Conversion optionnelle en millimetres via un facteur de calibration
- Discussion explicite des limites physiques (perspective, resolution, bounding box vs fissure reelle)

## Architecture

```
Image beton  -->  YOLO11n (inference)  -->  Bounding Boxes (x, y, w, h, conf)
                                                    |
                                                    v
                                        Extraction metrologique
                                       (surface px2, diag, ratio w/h)
                                                    |
                                           +--------+--------+
                                           |                 |
                                           v                 v
                                   Image annotee     Tableau Pandas
                                  (boxes + labels)   (mesures/fissure)
                                           |                 |
                                           v                 v
                                      Interface Streamlit (dashboard)
```

## Resultats

| Metrique | Valeur |
|---|---|
| mAP@0.5 | > 0.85 (apres fine-tuning) |
| mAP@0.5:0.95 | > 0.50 |
| Inference CPU | ~60 ms/image (YOLO11n) |
| Parametres modele | 2.6M (nano) |

*Les valeurs exactes dependent du dataset et du nombre d'epochs.*

## Dataset

**Concrete Crack Detection** depuis Roboflow Universe -- Images reelles de surfaces beton avec fissures annotees en bounding boxes (~1 200 images).

Source : [Roboflow Universe - Concrete Crack](https://universe.roboflow.com/vijayalakshmi-2yshx/concrete-surface-crack-detection-using-yolov5-model)

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Detection | YOLO11n (Ultralytics) | Etat de l'art sept. 2024, anchor-free, 2.6M params |
| Framework | PyTorch (via Ultralytics) | API haut niveau, fine-tuning en 5 lignes |
| Dataset | Roboflow Universe | Format YOLO natif, images reelles annotees |
| Metrologie | NumPy + Pandas | Extraction et structuration des mesures |
| Interface | Streamlit | Upload + inference + tableau dimensionnel |

## Structure du projet

```
CrackVision/
├── README.md
├── requirements.txt
├── crack_vision.py              # Notebook Colab complet (fine-tuning + inference)
├── cours_crackvision.md         # Cours pedagogique (YOLO11, metrologie, pieges)
├── app.py                       # Interface Streamlit (genere par le notebook)
├── data/                        # Dataset (telecharge automatiquement)
│   ├── dataset.yaml
│   ├── train/
│   └── valid/
├── runs/                        # Resultats d'entrainement (genere)
│   └── detect/
│       └── crack_detection/
│           └── weights/
│               └── best.pt      # Meilleur modele
└── models/                      # Modeles exportes (optionnel)
```

## Installation et execution

```bash
# Cloner le repo
git clone https://github.com/<votre-username>/CrackVision.git
cd CrackVision

# Installer les dependances
pip install -r requirements.txt

# Option 1 : Notebook Colab (recommande pour le fine-tuning)
# Ouvrir crack_vision.py dans Google Colab (Runtime > GPU T4)

# Option 2 : Lancer l'interface Streamlit (apres entrainement)
streamlit run app.py
```

## Fonctionnalites de l'interface Streamlit

- **Upload** d'image de surface beton (jpg, png)
- **Inference** YOLO11 avec seuil de confiance ajustable
- **Affichage** de l'image annotee avec bounding boxes
- **Tableau metrologique** : pour chaque fissure, confiance, largeur, hauteur, surface, diagonale, ratio d'aspect
- **Calibration optionnelle** : saisie du facteur pixel/mm pour conversion en dimensions reelles

## Limites et extensions

### Limites physiques (discutees dans le cours)

1. **Bounding box vs fissure** : la boite englobante surestime la surface reelle, surtout pour les fissures diagonales
2. **Pas de mesure absolue** : sans facteur de calibration (objet de reference, distance connue), les dimensions restent en pixels
3. **Perspective** : si la camera n'est pas perpendiculaire, les mesures sont deformees
4. **Resolution** : les fissures fines disparaissent apres le resize a 640px

### Extensions possibles

- **YOLO11-seg** (segmentation d'instance) pour des masques pixel par pixel
- **Squelettisation** du masque pour mesurer l'ouverture de fissure (parametre critique en genie civil)
- **Tiling** : decouper les images haute resolution en tuiles pour detecter les fissures fines
- **Export ONNX** pour inference embarquee sur drone ou robot d'inspection
- **Rapport PDF** automatique avec localisation et dimensions de chaque defaut

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.
Profil : Docteur-Ingenieur, expertise physique/metrologie/CND, reconversion IA.

---

*Ce projet demontre la capacite a deployer un modele de detection etat de l'art (YOLO11) sur un probleme industriel reel, tout en apportant une reflexion metrologique que seul un profil ingenieur/physicien peut fournir.*
