# Autorités de certification supplémentaires

Si votre réseau inspecte le HTTPS (proxy d'entreprise, antivirus), `pip` échoue
dans les conteneurs avec une erreur de vérification TLS. Exportez le certificat
racine correspondant au format PEM avec l'extension `.crt` et déposez-le ici :
les Dockerfiles l'ajoutent aux autorités de confiance (`update-ca-certificates`)
sans désactiver la vérification.

Les fichiers `*.crt` de ce dossier sont ignorés par git.
