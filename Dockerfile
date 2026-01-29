# Image de base Python 3.8
FROM python:3.8-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet dans le container
COPY . .

# Commande par défaut au lancement du container
# Ouvre un shell bash interactif
CMD ["/bin/bash"]
