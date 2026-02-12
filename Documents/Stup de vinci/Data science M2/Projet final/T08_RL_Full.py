import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO # Choix de l'algorithme [cite: 139]

# 1. Définition de l'Environnement (State, Action, Reward) [cite: 133, 143]
class TradingEnv(gym.Env):
    # 1. Dans __init__, définit les colonnes à exclure une fois pour toutes
    def __init__(self, df, initial_balance=10000, fee=0.0001):
        super(TradingEnv, self).__init__()
        self.df = df
        self.exclude_cols = ['returns', 'open', 'high', 'low', 'close', 'target'] # Liste complète
        
        # Calcul dynamique du nombre de features réelles
        sample_obs = self.df.iloc[0].drop(labels=self.exclude_cols, errors='ignore')
        num_features = len(sample_obs)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.initial_balance = initial_balance
        self.fee = fee
        self.reset()

    # 2. Utilise la même logique dans reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        obs = self.df.iloc[self.current_step].drop(labels=self.exclude_cols, errors='ignore').values
        return obs.astype(np.float32), {}

    # 3. Et la même logique dans step
    def step(self, action):
        # Calcul reward
        try:
            price_return = self.df.iloc[self.current_step]['returns']
        except KeyError:
            price_return = 0 # Sécurité si colonne absente
            
        reward = 0
        if action == 1: # BUY
            reward = price_return - self.fee
        elif action == 2: # SELL
            reward = -price_return - self.fee
            
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        obs = self.df.iloc[self.current_step].drop(labels=self.exclude_cols, errors='ignore').values
        
        return obs.astype(np.float32), float(reward), done, False, {}

# 2. Chargement et Split (2022 pour l'entraînement) [cite: 25]
df = pd.read_parquet("data/GBPUSD_M15_FEATURES.parquet")
train_df = df[df.index.year == 2022]

# 3. Entraînement avec Paramètres clés [cite: 145]
env = TradingEnv(train_df)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, batch_size=64)
print("Début de l'entraînement RL...")
model.learn(total_timesteps=10000) # À augmenter selon tes résultats

# 4. Sauvegarde du modèle [cite: 187]
model.save("models/v1/ppo_gbpusd_m15")
print("Modèle sauvegardé dans models/v1/")