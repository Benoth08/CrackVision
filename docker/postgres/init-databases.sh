#!/bin/bash
# Cree une base et un utilisateur par service (Airflow, MLflow) au premier demarrage.
set -euo pipefail

create_database() {
    local name="$1" password="$2"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
        CREATE USER ${name} WITH PASSWORD '${password}';
        CREATE DATABASE ${name} OWNER ${name};
SQL
}

create_database airflow "${AIRFLOW_DB_PASSWORD}"
create_database mlflow "${MLFLOW_DB_PASSWORD}"
