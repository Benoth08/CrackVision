# Journal des décisions

Chaque choix technique important du passage de la version 1 (notebook Colab
YOLO11) à la version 2 (YOLOX, autoencodeur, hybride, pile MLOps), avec sa raison.
Les points encore ouverts sont regroupés en fin de document.

## Données

**D1. Jeu de données : Concrete Crack Segmentation Dataset (Özgenel, 2019).**
458 photos 4032 × 3024 de murs en béton (campus METU) avec leurs masques de
fissures, licence CC BY 4.0, téléchargeables sans compte avec une empreinte
SHA-256 publiée. Le jeu Roboflow de la version 1 exige une clé d'API et ne
contient pas de zones saines du même domaine.
Ce jeu est la source du célèbre jeu de classification de 40 000 vignettes
227 × 227 du même auteur. Je ne l'ai pas utilisé : ses vignettes ne disent pas
de quelle photo elles viennent, ce qui interdit un découpage sans fuite.

**D2. Données saines pour l'autoencodeur : les tuiles sans fissure des mêmes photos.**
Les fissures couvrent en moyenne 1,4 % des pixels, mais elles sont longues et
traversent plusieurs tuiles : environ la moitié des tuiles sont saines (1 827 sur
3 840 en entraînement). Une tuile est déclarée saine seulement si aucun pixel de
fissure ne se trouve dans la tuile ni dans une marge de 16 px autour d'elle.
Mélanger deux sources (photos fissurées d'un jeu, béton sain d'un autre) aurait
appris au modèle à reconnaître la source plutôt que la fissure.

**D3. Tuiles de 512 px après réduction de moitié (2016 × 1512).**
Ce choix donne 12 tuiles par photo et la résolution native de YOLOX en entrée
sans redimensionnement. Les fissures fines (2 à 10 px après réduction) restent
visibles. Les masques JPEG sont binarisés à pleine résolution avant réduction,
puis un pixel est gardé s'il est couvert à plus de 25 %, pour que les fissures
fines ne disparaissent pas.

**D4. Boîtes englobantes issues des composantes connexes du masque**, après une
fermeture morphologique qui recolle les fragments d'une même fissure. Une
composante de moins de 60 px est ignorée. Les tuiles qui ne contiennent qu'un tel
fragment, ou dont une fissure frôle le bord, sont « ambiguës » : elles sont
exclues de tous les jeux et comptées dans les rapports.

**D5. Découpage stratifié par photo : 70 % entraînement, 15 % validation, 15 % blind test.**
Les tuiles d'une même photo ne sont jamais réparties entre deux jeux, pour éviter
une fuite d'information. Les strates croisent le taux de fissure (3 quantiles) et
le nombre de fissures (1, 2-3, 4 et plus). L'algorithme parcourt les strates
l'une après l'autre et affecte chaque photo au jeu le plus en retard sur sa
proportion cible. Il est déterministe, avec la graine 42.

**D6. Jeu d'entraînement insensible au déséquilibre**, sur deux niveaux.
Mesure faite sur le jeu réel : à l'échelle des tuiles, les classes sont presque
équilibrées (49 % fissurées, 48 % saines, 3 % ambiguës). Le déséquilibre porte
surtout sur les sous-types de fissures. Les deux corrections restent actives, car
la proportion de tuiles saines dépend de la taille de tuile et d'un autre jeu de
photos.
- dans les tuiles fissurées : *repeat factor sampling* (LVIS) sur la taille, l'orientation
  et le nombre de fissures ; une tuile d'une strate rare est répétée √(t/f) fois en moyenne ;
- entre classes : on tire des tuiles saines jusqu'à 50 % de tuiles fissurées.
Les répétitions sont écrites dans l'annotation COCO (nouvel identifiant, même
fichier) : aucune image n'est copiée. La validation et le blind test gardent la
distribution naturelle.

**D7. Blind test scellé.** Une empreinte SHA-256 de chaque fichier est écrite à
sa création (`MANIFEST.sha256`). Elle est revérifiée avant l'évaluation et
enregistrée dans MLflow. Seule l'étape `evaluate@blind_test` en dépend, ce que
vérifie un test (`tests/test_dvc_pipeline.py`). Les seuils et la fusion sont
fixés sur la validation uniquement.

## Modèles

**M1. YOLOX-s (Megvii, Apache 2.0) au commit `6ddff48`, pré-entraîné sur COCO.**
Je l'ai figé par empreinte. La licence Apache 2.0 lève la contrainte AGPL
d'Ultralytics. La variante s (9 M de paramètres) est un bon compromis à 512 px ;
`nano` et `tiny` restent disponibles par `params.yaml`.

**M2. Boucle d'entraînement indépendante du matériel.** Le Trainer officiel de
YOLOX ne fonctionne qu'avec CUDA (`torch.cuda.set_device`, évaluateur en
`torch.cuda.FloatTensor`). La boucle réutilise l'Exp officielle (jeu COCO,
mosaïque, optimiseur, LR *yoloxwarmcos*, EMA, précision mixte sur GPU) mais tourne
aussi sur CPU. Cela permet un pipeline de fumée complet en intégration continue.

**M3. Sélection du meilleur YOLOX sur l'AP50 de validation.** Pour une fissure
fine et allongée, l'AP50:95 pénalise surtout l'imprécision des bords de boîte,
qui n'a pas de sens physique ici. Les deux sont rapportés.

**M3 bis. Constat sur la mosaïque (mesuré sur la validation, entraînement du 23/10).**
Pendant les 50 époques avec mosaïque, l'AP50 de validation plafonne vers 0,64.
Elle bondit à 0,86 dès que la mosaïque est coupée pour les 10 dernières époques
(AP50:95 de 0,28 à 0,68). La mosaïque assemble quatre tuiles réduites : les
fissures y deviennent plus fines et plus petites qu'au test. Une piste à mesurer
sur la validation : allonger la phase sans mosaïque, ou ramener `mosaic_scale`
vers (0,8 ; 1,2).

**M4. Autoencodeur convolutif débruiteur (entrée 256 px).** Son architecture
(profondeur 5, goulot 8 × 8 × 8) a été réglée sur la validation (voir M4 bis).
La perte mêle L1 et 1 − SSIM ; la carte d'anomalie utilise la même combinaison,
lissée par un filtre gaussien. Le score d'une tuile est la moyenne du top 1 % de
sa carte. La normalité est calibrée sans étiquette, sur les tuiles saines de
validation : moyenne, écart-type et quantiles 99 / 99,5 / 99,9 %.

**M4 bis. Réglage de l'autoencodeur après un premier passage complet (23/10).**
Le premier autoencodeur (profondeur 4, goulot 16 × 16 × 32, 80 époques) avait été
retenu sur la plus faible perte de reconstruction des tuiles saines de validation.
Son AUROC par tuile ne valait que 0,80 en validation et 0,82 sur le blind test.
Un réglage court, **sur la validation uniquement**, a montré deux choses :
- le critère de sélection était en cause. Le même réseau arrêté à 40 époques
  atteint 0,951 d'AUROC. À force d'apprendre à reconstruire le béton, le réseau
  finit par reconstruire aussi les fissures, et l'anomalie disparaît. Le point de
  contrôle est désormais choisi sur l'AUROC de la validation annotée : le modèle
  reste entraîné sur des tuiles saines uniquement, et la validation sert, comme le
  recommande l'article, de corpus de statut connu ;
- un goulot plus profond localise bien mieux.

| Configuration (40 époques, validation) | AUROC tuile | AUROC pixel | AP pixel |
|---|---|---|---|
| profondeur 4, goulot 16 × 16 × 32 | 0,951 | 0,946 | 0,34 |
| profondeur 4, goulot 16 × 16 × 8 | 0,918 | 0,911 | 0,19 |
| profondeur 5, goulot 8 × 8 × 32 | 0,953 | 0,985 | 0,75 |
| **profondeur 5, goulot 8 × 8 × 8, bruit 0,1 (retenue)** | **0,952** | **0,986** | **0,80** |

La configuration retenue est à égalité sur le critère principal et localise
mieux. Son goulot plus étroit la rend aussi moins apte à reconstruire un défaut
inconnu. Seuls l'autoencodeur, la fusion et les évaluations ont été refaits (DVC) ;
YOLOX n'a pas changé. Résultats du premier passage sur le blind test, pour
mémoire : AUROC de l'autoencodeur 0,817, F1 du modèle hybride 0,978.

**M5. Modèle hybride : deux régressions logistiques ajustées sur la validation.**
- Au niveau de la boîte, chaque détection YOLOX est repondérée par l'évidence de
  l'autoencodeur à l'intérieur de la boîte. Le modèle apprend sur les détections
  de validation étiquetées vrai ou faux positif à IoU 0,5.
- Au niveau de la tuile, la probabilité de défaut combine le meilleur score
  YOLOX, le nombre de boîtes, le score d'anomalie centré-réduit et la surface anormale.
- Le filet de sécurité remonte comme « anomalie non classée » toute zone au-delà
  du quantile 99,9 % des pixels sains qui ne recouvre aucune boîte.
Les classes sont pondérées de façon équilibrée. Les seuils sont choisis sur des
prédictions hors échantillon (validation croisée groupée par photo). C'est
l'architecture recommandée par l'article de référence : le supervisé couvre le
volume, le non supervisé couvre le risque.

**M6. Seuils de décision au F1 maximal sur la validation**, pour les trois
modèles. Les métriques sans seuil (AUROC, AP) sont rapportées à côté. Le MCC et
l'exactitude équilibrée, peu sensibles au déséquilibre, aussi.

## Évaluation

**E1. Intervalles de confiance à 95 % par bootstrap groupé par photo**
(2 000 rééchantillonnages, graine fixe). Les tuiles d'une même photo ne sont
pas indépendantes : rééchantillonner des tuiles aurait donné des intervalles
trop étroits.

**E2. AP COCO officielle (pycocotools) pour les valeurs ponctuelles**, et une AP50
recalculée de la même manière (interpolation sur 101 points) pour le bootstrap,
1 000 fois plus rapide.

## Industrialisation

**I1. Microservices Docker** : `api` (FastAPI), `nginx`, `mlflow` (+ PostgreSQL),
`prometheus`, `grafana`, `nginx-exporter`, Airflow 3.2.1 (api-server, scheduler,
dag-processor, LocalExecutor), et `trainer`. L'image `trainer` exécute les
étapes et Airflow la lance par DockerOperator.

**I2. Une seule image de base pour `trainer` et `api`** (Dockerfile multi-cibles).
La roue YOLOX, avec son extension C++ COCOeval, est compilée une fois dans une
étape de construction. L'API tourne avec un utilisateur non root. PyTorch est en
version CPU par défaut ; l'argument de construction `TORCH_VARIANT=cuda`
(variable `TRAINER_TORCH` de `.env`) donne une image `trainer` pour GPU NVIDIA.

**I3. DVC orchestre les données et les modèles ; Airflow orchestre DVC.** Chaque
tâche Airflow exécute `dvc repro --single-item <étape>`, et DVC saute ce qui est à
jour. Les tâches DVC partagent un pool d'une seule place : DVC verrouille le
dépôt, deux `dvc repro` simultanés échoueraient. Un mode `smoke` du DAG rejoue les
mêmes commandes sans DVC, sur des données synthétiques.

**I4. Remote DVC local `.dvc-storage/`** (ignoré par git). Le remplacer par S3 ou
MinIO en production se fait en une commande (`dvc remote modify`).

**I5. Suivi MLflow en deux temps.** Sur un GPU distant sans accès au serveur
MLflow, l'entraînement écrit son historique dans les dossiers des modèles. La
commande `crackvision publish` le rejoue ensuite dans MLflow : un run parent, un
run par modèle, et l'enregistrement des trois modèles au format pyfunc. Quand le
serveur est joignable pendant l'entraînement, le suivi est en direct.

**I6. Promotion contrôlée.** L'alias `champion` n'est accordé que si le blind
test passe les seuils de `promotion.gates` et fait au moins aussi bien que le
champion actuel sur le même blind test (même empreinte). Sinon, l'alias est
`challenger`. L'API charge le modèle hybride `champion`, qui embarque YOLOX et
l'autoencodeur : le trio servi est toujours cohérent. Un modèle de fumée est
publié sous d'autres noms et ne peut pas devenir champion.

**I7. nginx, point d'entrée unique** (port 8090). `/api/metrics` et `/api/admin/`
n'y sont pas exposés. `/api/predict` a une limite de débit. Les amonts sont
résolus à la demande, si bien que nginx démarre même si un service optionnel est
arrêté. Airflow a son propre port (8091) : son interface sous un sous-chemin
demanderait de réécrire l'URL de l'API d'exécution interne, une source de pannes
pour un gain nul.

**I8. Surveillance de la dérive sans étiquettes.** Le score d'anomalie de
l'autoencodeur, centré-réduit sur le béton sain de validation, sert
d'indicateur. Une moyenne durablement supérieure à 2 signale des images qui ne
ressemblent plus aux données d'entraînement. S'y ajoutent la luminosité des
entrées et la part d'images défectueuses. Une alerte Prometheus et un DAG
Airflow horaire l'exploitent. Ce DAG peut déclencher un réentraînement si la
Variable `crackvision_auto_retrain` vaut `true`.

**I9. Autorités de certification supplémentaires dans les images** (`docker/certs/`).
Derrière un proxy d'entreprise ou un antivirus qui inspecte le HTTPS, `pip` échoue
dans les conteneurs. Le certificat racine se dépose dans ce dossier, ignoré par
git, sans désactiver la vérification TLS.

**I10. Versions figées.** Les versions sont mûres et testées ensemble (torch 2.5.1,
numpy 1.26, opencv 4.10, MLflow 3.4…), plutôt que les dernières parues. Le code
de YOLOX date de 2022-2023 et n'est pas garanti avec numpy 2.x ou opencv 5.

**I10 bis. Trois dépendances transitives figées après des pannes constatées** :
- `pathspec==0.12.1` : `dvc 3.59.1` importe `_DIR_MARK`, que pathspec 1.x a retiré ;
- `sqlalchemy==2.0.36` : MLflow 3.4 importe `FallbackAsyncAdaptedQueuePool`, que les
  versions récentes ont retiré ;
- l'image MLflow installe ses paquets avec `uv` plutôt que `pip`. Derrière un
  antivirus qui inspecte le HTTPS, une réponse compressée de PyPI arrivait
  tronquée (`InvalidChunkLength`) et `pip` échouait à chaque essai ; `uv` passe.

**I10 ter. Adresse de suivi MLflow : racine du serveur, sans préfixe.**
Avec `--static-prefix=/mlflow`, MLflow 3.4 ne préfixe que l'interface et ses routes
`ajax-api`. L'API REST et `/health` restent à la racine. Les clients (API, trainer,
Airflow) visent donc `http://mlflow:5000`, et nginx sert l'interface sous `/mlflow/`.

**I11. L'ancien script `crack_vision.py` (YOLO11) est retiré de l'arborescence.**
Il reste dans le premier commit de l'historique git. Il ne s'exécutait plus en
l'état : une définition de fonction et le début d'une chaîne y manquaient.

## Vérifications effectuées

| Élément | Où | Résultat |
|---|---|---|
| Suite de tests (51 tests, dont le pipeline de fumée complet, l'API et la galerie) | machine GPU louée, environnement identique à l'image `trainer` | 51 réussis |
| Tests rapides (37) et `ruff` | image Docker `trainer` | réussis, aucune alerte |
| Construction des images `trainer`, `api`, `mlflow`, `airflow` | Docker | réussie |
| Import des deux DAG dans l'image Airflow 3.2.1 | Docker | aucune erreur |
| Fichier Compose (12 services) et configuration nginx | `docker compose config`, `nginx -t` sur l'image nginx 1.27 | valides |
| Pipeline DVC complet sur les données réelles | machine GPU louée (A40) | `dvc repro` puis `dvc push` réussis |
| Galerie de six photos entières du blind test (`crackvision gallery`) | machine GPU louée (A4500) | réussie, 53 s |
| PostgreSQL + script d'initialisation, MLflow 3.4, `crackvision publish`, API chargée depuis le registre, nginx (configuration du dépôt), nginx-exporter, Prometheus (3 cibles, 6 règles), Grafana provisionné, Airflow (import, 16 tâches, DAG de dérive exécuté) | machine GPU louée, services lancés sans Docker | toutes les routes répondent ; 114 prédictions sans erreur |

**Ce qui n'a pas encore été vérifié** : la pile complète lancée par `docker compose up` d'un
seul tenant. Les services ont été vérifiés un par un, avec les fichiers de configuration du
dépôt, sur une machine GPU louée qui ne peut pas lancer de conteneurs. Une seule adaptation y a
été nécessaire : les noms d'hôte Docker (`api`, `mlflow`…) remplacés par `127.0.0.1` dans une
copie de `nginx.conf`. Pour la même raison, le DAG d'entraînement n'a pas été exécuté de bout en
bout par DockerOperator ; ses commandes sont celles que DVC et le pipeline de fumée ont
exécutées. L'image `trainer` en variante CUDA (`TRAINER_TORCH=cuda`) n'a pas été construite.

## Points ouverts

- Les masques d'origine ne contiennent pas toutes les fissures capillaires : en
  relisant les figures, on voit des tuiles « saines » qui portent un trait de
  fissure très fin (par exemple 516_r0_c3 ou 416_r0_c2 dans l'entraînement). Une
  partie des fausses alarmes mesurées sur les tuiles saines vient de ces oublis
  d'annotation. Les métriques sont donc légèrement pessimistes.
- Le jeu ne contient que des fissures : la capacité du filet de sécurité à
  remonter d'autres défauts (épaufrures, taches, armatures apparentes) n'est pas
  mesurée faute d'exemples annotés.
- Les mesures de longueur et d'ouverture par pseudo-masque sont indicatives :
  la carte d'anomalie lissée élargit la fissure. Une vraie segmentation serait
  nécessaire pour une métrologie d'ouverture fiable.
