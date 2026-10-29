# CrackVision — Détection de fissures dans le béton

Trois modèles pour trouver les fissures sur des photos de murs en béton :

- **YOLOX-s** (supervisé) : détecteur qui dessine une boîte autour de chaque fissure.
- **Autoencodeur** (non supervisé) : entraîné **uniquement sur du béton sain**. Ce qu'il reconstruit mal est une anomalie.
- **Hybride** : régression logistique qui combine les deux, au niveau de la tuile et de chaque boîte.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOX](https://img.shields.io/badge/Détection-YOLOX--s-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c)
![MLOps](https://img.shields.io/badge/MLOps-Airflow%20·%20MLflow%20·%20DVC-0a7)
![Monitoring](https://img.shields.io/badge/Monitoring-Prometheus%20·%20Grafana-e6522c)

![Inférence du modèle hybride sur une photo du blind test](docs/images/blind_full_photo.jpg)

---

## Résultats

Blind test de **69 photos** (810 tuiles de 512 px, dont 404 fissurées). Seuils fixés sur la validation.

| | YOLOX | Autoencodeur | Hybride |
|---|---|---|---|
| Tuile fissurée ou saine — F1 | 0,977 | 0,901 | **0,978** |
| Tuile fissurée ou saine — AUROC | **0,997** | 0,957 | 0,994 |
| Détection des boîtes — AP50 | **0,868** | — | **0,868** |
| Localisation au pixel — AUROC | — | **0,977** (sans étiquette) | — |
| Latence par tuile (GPU) | 5,4 ms | 2,5 ms | 9,6 ms |

### Exemples

Vérité terrain (masque et boîtes en pointillés), puis sortie de chaque modèle. En orange, la carte d'anomalie de l'autoencodeur.

![Exemples du blind test](docs/images/blind_examples.jpg)

### Galerie : photos entières du blind test

À gauche la vérité terrain, à droite le modèle hybride : boîtes en vert, anomalie en orange, anomalies non classées en violet.

![Galerie, photos 490, 027 et 445](docs/images/galerie_1.jpg)

![Galerie, photos 498, 555 et 083](docs/images/galerie_2.jpg)

### Test de résistance : défauts jamais vus

Quatre défauts synthétiques (tache d'humidité, efflorescence, rouille, éclat) incrustés dans 100 tuiles saines. Aucun modèle ne les signale au-delà de son taux de fausses alarmes.

![Défauts hors classe](docs/images/stress_examples.jpg)

![Taux de signalement des défauts hors classe](docs/images/stress_rates.png)

### Entraînement

![Courbes d'entraînement](docs/images/training_curves.png)

### Données

Découpage **par photo** : aucune photo n'est partagée entre train, validation et blind test.

![Une photo et son masque](docs/images/photo_and_mask.png)

![Répartition des tuiles](docs/images/split_distribution.png)

![Rééquilibrage](docs/images/balance.png)

---

## MLOps

| Outil | Rôle |
|---|---|
| **DVC** | versionne données et modèles ; `dvc repro` ne relance que les étapes modifiées |
| **MLflow** | suivi des runs, registre ; l'alias `champion` n'est donné que si le blind test passe les seuils (AP50 ≥ 0,30, AUROC AE ≥ 0,70, F1 hybride ≥ 0,70) **et** fait au moins aussi bien que le champion en place |
| **Airflow** | DAG d'entraînement (chaque étape DVC dans un conteneur) + DAG de dérive qui interroge Prometheus chaque heure et peut relancer un entraînement |
| **FastAPI + nginx** | API d'inférence et de métrologie derrière une passerelle unique |
| **Prometheus + Grafana** | latence, erreurs, et dérive **sans étiquette** : score d'anomalie centré-réduit, luminosité, part d'images signalées |
| **Docker Compose** | toute la pile en microservices |

### Airflow

![DAG d'entraînement dans Airflow](docs/images/capture_airflow.png)

### MLflow

<p align="center">
  <img src="docs/images/capture_mlflow_registre.png" width="49%" alt="Registre MLflow, alias champion">
  <img src="docs/images/capture_mlflow_courbes.png" width="49%" alt="MLflow, AP de validation de YOLOX par époque">
</p>

### Grafana

![Tableau de bord Grafana](docs/images/capture_grafana.png)

### Alertes Prometheus

![Règles d'alerte Prometheus](docs/images/capture_prometheus.png)

### Interface de démonstration

![Interface de démonstration](docs/images/capture_interface.jpg)

---

## Démarrage rapide

```bash
cp .env.example .env
docker compose --profile tools --profile airflow build
docker compose up -d
docker compose run --rm trainer python -m pytest -q                # vérif. sans GPU ni données
docker compose run --rm trainer dvc repro                          # données + modèles (GPU conseillé)
docker compose run --rm trainer python -m crackvision.cli publish  # MLflow + alias champion
docker compose --profile airflow up -d                             # orchestration (facultatif)
```

| Service | Adresse |
|---|---|
| Interface et API | http://localhost:8090 · `/api/docs` |
| MLflow, Grafana, Prometheus | `/mlflow/`, `/grafana/`, `/prometheus/` |
| Airflow | http://localhost:8091 |

*Les choix détaillés sont dans [DECISIONS.md](DECISIONS.md).*
