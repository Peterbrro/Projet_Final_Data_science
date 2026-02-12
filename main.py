import os
import pandas as pd
from stable_baselines3 import PPO

# Import de nos modules personnalisés
from utils import task_01_import_m1, task_02_03_aggregate_and_clean
from features import calculate_features
from engine import task_04_eda, task_06_baselines, task_07_ml, task_09_eval_robuste
from trading_env import TradingEnv

if __name__ == "__main__":
    # 1. Setup
    DATA_DIR, OUT_DIR = 'data/', 'output/'
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 2. Pipeline de données (T01-T03)
    df_raw = task_01_import_m1(DATA_DIR)
    df_m15 = task_02_03_aggregate_and_clean(df_raw)

    # 3. EDA (T04)
    df_eda = task_04_eda(df_m15, OUT_DIR)

    # 4. Features (T05)
    df_full = calculate_features(df_eda)

    # 5. Split Temporel Strict (Exigence Projet)
    df_train = df_full[df_full['timestamp'].dt.year <= 2023].copy()
    df_test_2024 = df_full[df_full['timestamp'].dt.year == 2024].copy()
    print(f"\nDonnées prêtes. Train: {len(df_train)} lignes | Test 2024: {len(df_test_2024)}")

    # 6. Baselines (T06)
    task_06_baselines(df_test_2024, OUT_DIR)

    # 7. Machine Learning (T07)
    model_ml = task_07_ml(df_train, df_test_2024, OUT_DIR)

    # 8. Reinforcement Learning (T08 & T09)
    print("\n--- T08 : ENTRAÎNEMENT RL ---")
    env_train = TradingEnv(df_train)
    model_rl = PPO("MlpPolicy", env_train, verbose=0)
    model_rl.learn(total_timesteps=30000) # Augmente si besoin
    model_rl.save("models/ppo_gbpusd_v1")

    # 9. Évaluation Finale (T09)
    task_09_eval_robuste(df_test_2024, model_rl, OUT_DIR)

    print(f"\nPipeline terminé. Résultats disponibles dans /{OUT_DIR}")