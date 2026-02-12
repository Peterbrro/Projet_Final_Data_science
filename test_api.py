import requests
import json
from datetime import datetime, timedelta

# L'adresse de ton Docker
URL = "http://localhost:8000/predict"

# 1. Génération de 210 bougies fictives (pour être sûr d'avoir de la marge)
data = []
start_time = datetime.now() - timedelta(days=3)

for i in range(210):
    candle_time = start_time + timedelta(minutes=15 * i)
    data.append({
        "timestamp": candle_time.strftime("%Y-%m-%d %H:%M:%S"),
        "open": 1.2700 + (i * 0.0001),
        "high": 1.2705 + (i * 0.0001),
        "low": 1.2695 + (i * 0.0001),
        "close": 1.2702 + (i * 0.0001),
        "vol": 1000 + i
    })

# 2. Envoi de la requête POST
print(f"🚀 Envoi de {len(data)} bougies à l'API...")
response = requests.post(URL, json=data)

# 3. Affichage du résultat
if response.status_code == 200:
    print("Signal reçu :")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Erreur {response.status_code}:")
    print(response.text)