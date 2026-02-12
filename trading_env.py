import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.features = ['return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
                         'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
                         'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14']
        
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self):
        obs = self.df.iloc[self.current_step][self.features].values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        price_change = self.df.iloc[self.current_step]['next_return']
        
        reward = 0
        if action == 1: reward = price_change      # Long
        elif action == 2: reward = -price_change   # Short
            
        done = self.current_step >= len(self.df) - 2
        return self.get_observation(), reward, done, False, {}