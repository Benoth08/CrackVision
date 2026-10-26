"""Metriques Prometheus exposees sur /metrics.

Au-dela du trafic (debit, erreurs, latence), on suit ce qui signale une derive :
distribution du score d'anomalie de l'autoencodeur (compare a sa valeur sur le
beton sain de validation, publiee en jauge), de la probabilite hybride, de la
luminosite des images recues, et la part d'images declarees defectueuses.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("crackvision_requests_total", "Requetes recues",
                   ["endpoint", "model", "status"])
REQUEST_LATENCY = Histogram("crackvision_request_latency_seconds",
                            "Duree de traitement d'une requete de prediction", ["model"],
                            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
INFERENCE_LATENCY = Histogram("crackvision_inference_seconds", "Duree de l'inference seule",
                              ["model"], buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
DECISIONS = Counter("crackvision_decisions_total", "Decisions rendues", ["model", "level"])
DETECTIONS = Histogram("crackvision_detections_per_image", "Fissures detectees par image",
                       ["model"], buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34))
CONFIDENCE = Histogram("crackvision_detection_confidence", "Confiance des fissures retenues",
                       ["model"], buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)))
# Score brut (toujours positif) : un histogramme Prometheus ne doit pas recevoir de
# valeurs negatives, sa somme serait prise pour un compteur remis a zero.
ANOMALY_SCORE = Histogram("crackvision_anomaly_score",
                          "Score d'anomalie de l'autoencodeur pour l'image",
                          buckets=(0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0))
# Reference : scores des tuiles saines de validation ; la derive se lit en
# (moyenne observee - mean) / std.
ANOMALY_REFERENCE = Gauge("crackvision_anomaly_reference",
                          "Score d'anomalie des tuiles saines de validation", ["stat"])
HYBRID_PROBA = Histogram("crackvision_hybrid_probability", "Probabilite de defaut (hybride)",
                         buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)))
BRIGHTNESS = Histogram("crackvision_input_brightness", "Luminosite moyenne des images (0-255)",
                       buckets=(25, 50, 75, 100, 125, 150, 175, 200, 225, 255))
MEGAPIXELS = Histogram("crackvision_input_megapixels", "Taille des images recues",
                       buckets=(0.1, 0.3, 1, 2, 4, 8, 12, 16, 24, 40))
UNKNOWN_REGIONS = Counter("crackvision_unknown_regions_total",
                          "Zones anormales sans boite YOLOX (filet de securite)")
MODEL_INFO = Gauge("crackvision_model_info", "Modele servi (valeur 1)",
                   ["registry_model", "version", "source"])
MODEL_READY = Gauge("crackvision_model_ready", "1 si les modeles sont charges")
RELOADS = Counter("crackvision_model_reloads_total", "Rechargements de modeles", ["status"])
