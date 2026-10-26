# syntax=docker/dockerfile:1.7
# =============================================================================
# Images Python de CrackVision (une cible par microservice)
#   - trainer : pipeline DVC, entrainement, evaluation, tests (lance par Airflow)
#   - api     : service FastAPI d'inference (utilisateur non root)
# Construction : docker compose build trainer api
# =============================================================================
ARG PYTHON_IMAGE=python:3.12-slim
ARG YOLOX_COMMIT=6ddff4824372906469a7fae2dc3206c7aa4bbaee
# cpu (defaut, api et trainer) ou cuda (trainer sur GPU NVIDIA, PyTorch CUDA 12.4)
ARG TORCH_VARIANT=cpu

# --- base : Python + autorites de certification supplementaires --------------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
# Proxy TLS d'entreprise ou antivirus qui inspecte le HTTPS : deposer son
# certificat racine (.crt) dans docker/certs/ (dossier ignore par git).
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# --- ml-core : PyTorch (CPU ou CUDA) + dependances communes -------------------
FROM base AS ml-core
ARG TORCH_VARIANT
COPY requirements/torch-${TORCH_VARIANT}.txt requirements/base.txt /tmp/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements/torch-${TORCH_VARIANT}.txt -r /tmp/requirements/base.txt

# --- builder : roue YOLOX (extension C++ COCOeval compilee) ------------------
FROM ml-core AS builder
ARG YOLOX_COMMIT
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install setuptools==75.6.0 wheel==0.45.1 ninja==1.11.1.3 \
    && pip wheel --no-deps --no-build-isolation -w /wheels \
       "https://github.com/Megvii-BaseDetection/YOLOX/archive/${YOLOX_COMMIT}.zip"

# --- ml : base commune avec YOLOX installe ------------------------------------
FROM ml-core AS ml
COPY --from=builder /wheels /wheels
RUN pip install --no-deps /wheels/*.whl && rm -rf /wheels

# --- trainer -----------------------------------------------------------------
FROM ml AS trainer
RUN apt-get update && apt-get install -y --no-install-recommends git libarchive-tools \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/ /tmp/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements/trainer.txt
COPY pyproject.toml README.md /opt/crackvision/
COPY src /opt/crackvision/src
RUN pip install --no-deps /opt/crackvision
# Le depot est monte dans /workspace : son code prime sur la copie de l'image,
# pour que DVC execute exactement les fichiers qu'il suit.
ENV PYTHONPATH=/workspace/src \
    CRACKVISION_PROJECT_ROOT=/workspace \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=*
WORKDIR /workspace
CMD ["python", "-m", "crackvision.cli", "--help"]

# --- api -----------------------------------------------------------------------
FROM ml AS api
COPY requirements/api.txt /tmp/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements/api.txt
COPY pyproject.toml README.md /opt/crackvision/
COPY src /opt/crackvision/src
COPY params.yaml /opt/crackvision/params.yaml
RUN pip install --no-deps /opt/crackvision \
    && useradd --create-home --uid 10001 crackvision
USER crackvision
ENV CRACKVISION_PROJECT_ROOT=/opt/crackvision \
    MODEL_CACHE_DIR=/home/crackvision/model-cache \
    MODELS_DIR=/opt/crackvision/models \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /home/crackvision
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "crackvision.serving.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--root-path", "/api", "--proxy-headers", "--forwarded-allow-ips", "*"]
