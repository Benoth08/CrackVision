# syntax=docker/dockerfile:1.7
# Serveur MLflow : suivi des experiences + registre de modeles.
# Metadonnees dans PostgreSQL, artefacts servis par le serveur (volume dedie).
FROM ghcr.io/astral-sh/uv:0.9.30 AS uv

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=1
# uv plutot que pip : sa pile HTTP tolere les proxys TLS qui alterent les
# reponses compressees (constate avec pip derriere un antivirus).
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements/mlflow.txt /tmp/mlflow.txt
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r /tmp/mlflow.txt \
    && useradd --create-home --uid 10002 mlflow \
    && mkdir -p /mlflow/artifacts && chown -R mlflow /mlflow
USER mlflow
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"
# Les options (base, prefixe /mlflow derriere nginx) sont passees par docker-compose.yml
ENTRYPOINT ["mlflow", "server"]
