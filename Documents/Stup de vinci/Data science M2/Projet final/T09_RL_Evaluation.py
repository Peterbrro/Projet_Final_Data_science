import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from T08_RL_Full import TradingEnv # On réutilise ton environnement

# 1. Chargement des données de 2023 (Validation)
df = pd.read_parquet("data/GBPUSD_M15_FEATURES.parquet")
val_df = df[df.index.year == 2023].copy()

# 2. Chargement du modèle entraîné
model = PPO.load("models/v1/ppo_gbpusd_m15")
env = TradingEnv(val_df)

# 3. Simulation sur 2023
obs, _ = env.reset()
history = []

print("Lancement du test sur l'année 2023...")
for _ in range(len(val_df)-1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    history.append(reward)
    if done: break

# 4. Calcul des métriques financières
returns = np.array(history)
cum_profit = np.cumsum(returns)
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) # 96 bougies de 15m par jour
max_drawdown = np.max(np.maximum.accumulate(cum_profit) - cum_profit)
profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))

print(f"\n--- RÉSULTATS VALIDATION 2023 ---")
print(f"Profit Cumulé : {cum_profit[-1]:.2%}")
print(f"Sharpe Ratio : {sharpe:.2f}")
print(f"Max Drawdown : {max_drawdown:.2%}")
print(f"Profit Factor : {profit_factor:.2f}")