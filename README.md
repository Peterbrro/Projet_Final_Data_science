# Projet_Final_Data_science


PROJET_FINAL/  
├── data/  
│   └── gbpusd_m1.csv          # Données sources (T01)  
├── models/                    # Registry des modèles (T11)  
│   ├── v1/  
│   │   └── ppo_trading_model.zip  # Ton meilleur modèle (3.36%)  
│   └── v2/                    # Dossier prêt pour les futurs tests  
├── output/                    # Exports (backtests, logs, modèles temporaires)  
├── api.py                     # Serveur FastAPI (T10)  
├── Dockerfile                 # Configuration Image Docker (T12)  
├── engine.py                  # Logique d'entraînement RL (T08)  
├── features.py                # Calcul des indicateurs techniques (T05)  
├── main.py                    # Script principal (entraînement + ML) (T07)  
├── requirements.txt           # Liste des dépendances (Python-multipart, Torch...)  
├── trading_env.py             # Environnement Gym personnalisé (T08)  
├── utils.py                   # Fonctions chargement & agrégation (T01-T02)  
└── test_api.py                # Ton script de test des 200 bougies  
 
