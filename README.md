# Système de décision de trading GBP/USD
### Projet Final Data Science - Sup de Vinci 2026

Ce projet implémente une pipeline complète de trading algorithmique sur la paire **GBP/USD**, allant de la récupération de données brutes haute fréquence (M1) à une interface de décision industrialisée via Docker.

---

## Aperçu du Projet
L'objectif est de prédire les mouvements du marché à 15 minutes et d'optimiser les prises de position via un agent de Reinforcement Learning (PPO) entraîné sur les données historiques.

- **Données** : 2022 (Train), 2023 (Val), 2024 (Test final - Out of Sample).
- **Modèles** : Random Forest (V1) & Reinforcement Learning PPO (V2).
- **Industrialisation** : API FastAPI, Dashboard Streamlit, Conteneurisation Docker.



---

## Architecture du Dépôt
L'organisation du projet respecte les standards d'industrialisation (Phase 11 du sujet) :

```text
PROJET_FINAL/
├── data/                    # Données brutes (M1) et agrégées (M15)
├── output/                  # Fichiers traités et logs
│   └── gbpusd_final_features.csv  # Dataset enrichi utilisé par le dashboard
├── models/                  # Registre des modèles sauvegardés (Versioning)
│   ├── v1/                  # Modèle validé
│   └── v2/                  # Versions expérimentales
├── api.py                   # Serveur FastAPI (T10)
├── dashboard.py             # Interface Streamlit de présentation (T13)
├── engine.py                # Logique d'exécution des trades et calculs de performance
├── features.py              # Logique du Feature Pack V2 (T05)
├── main.py                  # Point d'entrée pour l'entraînement et l'orchestration
├── trading_env.py           # Environnement personnalisé Gymnasium pour le RL (T08)
├── utils.py                 # Fonctions d'agrégation M1->M15 et nettoyage
├── Dockerfile               # Configuration de l'image Docker (T12)
├── requirements.txt         # Dépendances Python (3.12 slim)
└── test_api.py              # Script de test des endpoints API
```

## Installation et Utilisation

Le projet est entièrement conteneurisé. L'environnement Python est pré-configuré dans l'image Docker pour garantir une exécution reproductible, sans conflit de dépendance locale.

### Déploiement via Docker

#### 1. Build de l'image
Cette étape compile l'environnement et installe les bibliothèques (FastAPI, Streamlit, Stable-Baselines3).
```powershell
docker build -t trading-app .
```
Accès navigateur : http://localhost:8501

#### 2. Lancer le Dashboard de présentation (T13)
L'interface visuelle permet de naviguer dans les résultats des différentes phases du projet.
```powershell
docker run -it --rm -p 8501:8501 -v ${PWD}/output:/app/output trading-app streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

#### 3. Lancer l'API de prédiction (T10)
Le serveur FastAPI expose le modèle de trading pour une consommation temps réel.

```powershell
docker run -it --rm -p 8000:8000 trading-app
```
Accès Swagger (API) : http://localhost:8000/docs

#### 4 Résultats Clés (Test Final 2024)

*Ces indicateurs sont calculés sur la période "Out of Sample" (données jamais vues par le modèle) pour garantir la robustesse de la stratégie.*

| Métrique | Résultat | Signification |
| :--- | :--- | :--- |
| **Profit RL** | **+3.36%** | Performance de l'agent PPO sur l'année 2024. |
| **Max Drawdown** | **-1.25%** | Risque maîtrisé avec une perte maximale historique faible. |
| **Profit Factor** | **1.42** | Ratio gains/pertes (stratégie robuste car > 1). |
| **Précision ML** | **52.4%** | Capacité du modèle Random Forest à prédire la direction. |

---
