# Utilise une version légère de Python
FROM python:3.12-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Copie le fichier des dépendances
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du projet (ton code + tes données)
COPY . .

# Commande par défaut au lancement du conteneur
CMD ["python", "main.py"]