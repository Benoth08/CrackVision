# syntax=docker/dockerfile:1.7
# Airflow 3 (LocalExecutor) : orchestre les etapes du pipeline dans des
# conteneurs "trainer" lances par DockerOperator.
FROM apache/airflow:3.2.1
USER root
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
USER airflow
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# Fournisseurs installes avec le fichier de contraintes officiel d'Airflow 3.2.1
RUN PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    && pip install --no-cache-dir "apache-airflow==3.2.1" \
       apache-airflow-providers-docker apache-airflow-providers-standard \
       --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.2.1/constraints-${PYTHON_VERSION}.txt"
