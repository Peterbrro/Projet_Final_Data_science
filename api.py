from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from stable_baselines3 import PPO
from features import calculate_features
import os

# 1. Définition du schéma de données pour l'API (Contrat T10)
class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    vol: float

app = FastAPI(
    title="Trading IA API - GBP/USD",
    description="API exposant le modèle de Reinforcement Learning (T10-T12)",
    version="1.0.0"
)

# 2. Configuration du chemin du modèle (Versioning T11)
# Note : On utilise le dossier monté via Docker
MODEL_PATH = "models/v1/ppo_trading_model.zip"

# Variable globale pour stocker le modèle en mémoire
model = None

@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'API."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = PPO.load(MODEL_PATH)
            print(f"Succès : Modèle V1 chargé depuis {MODEL_PATH}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
    else:
        print(f"⚠️ Erreur : Fichier {MODEL_PATH} introuvable. Vérifiez le montage Docker.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API de Trading IA opérationnelle.",
        "version": "v1",
        "endpoints": ["/predict", "/docs"]
    }

@app.post("/predict")
def predict(data: List[Candle]):
    """
    Endpoint de prédiction :
    Prend une liste de bougies (minimum 200 pour les calculs d'EMA200/RSI).
    Renvoie le signal : LONG, SHORT ou CASH.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur.")

    try:
        # 1. Conversion des données Pydantic en DataFrame
        df_input = pd.DataFrame([c.model_dump() for c in data])
        
        # Vérification de la quantité de données pour les indicateurs (T05)
        if len(df_input) < 200:
            return {
                "error": f"Données insuffisantes. Reçu: {len(df_input)} bougies. "
                         f"Le modèle nécessite au moins 200 bougies pour calculer l'EMA200."
            }

        # 2. Calcul des indicateurs techniques (Appel de features.py)
        df_with_features = calculate_features(df_input)
        
        # 3. Sélection des colonnes utilisées par l'agent RL
        features_list = [
            'return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
            'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
            'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14'
        ]
        
        # 4. Extraction du dernier état (dernière ligne)
        current_state = df_with_features[features_list].iloc[-1].values
        
        # 5. Prédiction avec le modèle PPO
        action, _ = model.predict(current_state, deterministic=True)
        
        # Traduction de l'action
        action_map = {0: "CASH (Rien)", 1: "LONG (Achat)", 2: "SHORT (Vente)"}
        
        return {
            "timestamp": str(df_with_features['timestamp'].iloc[-1]),
            "signal": action_map[int(action)],
            "action_code": int(action)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")