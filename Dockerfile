# Utilise une version légère de Python (on garde ta 3.12)
FROM python:3.12-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Installation de build-essential (parfois requis pour certaines libs de calcul)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie le fichier des dépendances
COPY requirements.txt .

# Installe les dépendances (vérifie que fastapi et uvicorn y sont !)
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du projet
COPY . .

# On expose le port 8000 pour accéder à l'API
EXPOSE 8000

# Commande par défaut : on lance l'API au lieu du script d'entraînement
# --host 0.0.0.0 est obligatoire pour que le port soit accessible hors du conteneur
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]