import pandas as pd
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# --- PHASE 8 : DÉFINITION DE L'ENVIRONNEMENT DE TRADING (RL) ---
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        
        # Actions : 0 = Cash (Rien), 1 = Long (Achat), 2 = Short (Vente)
        self.action_space = spaces.Discrete(3)
        
        # Observation : On utilise les colonnes de features calculées en T05
        # Il y a 16 features principales dans ta liste
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.get_observation()
        return obs, {}

    def get_observation(self):
        # On extrait les features pour le pas de temps actuel
        features = ['return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
                    'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
                    'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14']
        obs = self.df.iloc[self.current_step][features].values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Récompense basée sur le mouvement du prix à la bougie suivante
        price_change = self.df.iloc[self.current_step]['next_return']
        
        reward = 0
        if action == 1:   # Long
            reward = price_change
        elif action == 2: # Short
            reward = -price_change
            
        done = self.current_step >= len(self.df) - 2
        obs = self.get_observation()
        
        return obs, reward, done, False, {}

# --- TES FONCTIONS EXISTANTES (MODIFIÉES POUR LES CHEMINS) ---

def task_01_import_m1(data_path):
    print("--- DÉBUT T01 : IMPORTATION ET VÉRIFICATION ---")
    # Filtre strict sur le nom pour éviter de lire les outputs CSV
    csv_files = sorted(glob.glob(os.path.join(data_path, "DAT_MT_*.csv")))
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
    li = []
    for f in csv_files:
        df = pd.read_csv(f, names=cols, sep=',', index_col=False)
        li.append(df)
    full_df = pd.concat(li, axis=0, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['date'] + ' ' + full_df['time'], format='%Y.%m.%d %H:%M')
    full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'vol']]
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    print(f"T01 Terminée : {len(full_df)} lignes.")
    return full_df

def task_02_aggregate_m1_to_m15(df_m1):
    print("\n--- DÉBUT T02 : AGRÉGATION M1 -> M15 ---")
    df_resample = df_m1.set_index('timestamp').copy()
    df_m15 = df_resample.resample('15min', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    })
    df_m15 = df_m15.dropna(subset=['open']).reset_index()
    print(f"T02 Terminée : {len(df_m15)} bougies M15.")
    return df_m15

def task_03_clean_m15(df_m15):
    print("\n--- DÉBUT T03 : NETTOYAGE M15 ---")
    df_clean = df_m15.copy()
    df_clean = df_clean[(df_clean['open'] > 0) & (df_clean['high'] >= df_clean['low'])]
    print(f"T03 Terminée : {len(df_clean)} bougies propres.")
    return df_clean

def task_04_exploratory_analysis(df, out_path):
    print("\n--- DÉBUT T04 : ANALYSE EXPLORATOIRE ---")
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['returns'].dropna(), bins=100, kde=True)
    plt.title("Distribution des Rendements GBP/USD")
    plt.savefig(f"{out_path}/distribution_rendements.png")
    
    df['hour'] = df['timestamp'].dt.hour
    hourly_vol = df.groupby('hour')['returns'].std()
    plt.figure(figsize=(10, 6))
    hourly_vol.plot(kind='bar')
    plt.title("Volatilité par Heure (Saisonnalité)")
    plt.savefig(f"{out_path}/volatilite_horaire.png")

    print("Calcul du Test ADF...")
    adf_test = adfuller(df['returns'].dropna())
    print(f"Statistique ADF : {adf_test[0]:.4f}")
    print(f"p-value : {adf_test[1]:.4e}")
    
    plt.close('all')
    print("T04 Terminée : Graphiques sauvegardés.")
    return df

def task_05_feature_engineering(df):
    print("\n--- DÉBUT T05 : FEATURE ENGINEERING (V2) ---")
    df = df.copy()
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_diff'] = df['ema_20'] - df['ema_50']
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rolling_std_20'] = df['close'].rolling(window=20).std()
    df['range_15m'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['ema_200'] = ta.ema(df['close'], length=200)
    df['distance_to_ema200'] = df['close'] - df['ema_200']
    df['slope_ema50'] = df['ema_50'].diff(5)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rolling_std_100'] = df['close'].rolling(window=100).std()
    df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_100']
    
    df = pd.concat([df, ta.adx(df['high'], df['low'], df['close'], length=14)], axis=1)
    df = pd.concat([df, ta.macd(df['close'])], axis=1)

    df = df.dropna().reset_index(drop=True)
    print(f"T05 Terminée : {df.shape[1]} colonnes créées.")
    return df

def task_06_baseline_strategies(df, out_path):
    print("\n--- DÉBUT T06 : STRATÉGIES BASELINE ---")
    df = df.copy()
    df['next_return'] = df['close'].pct_change().shift(-1)
    
    df['strat_buy_hold'] = df['next_return']
    df['strat_random'] = np.random.choice([1, -1], size=len(df)) * df['next_return']
    
    df['signal_rsi'] = 0
    df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1
    df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1
    df['strat_rsi'] = df['signal_rsi'] * df['next_return']
    
    perf_bh = (1 + df['strat_buy_hold'].dropna()).prod() - 1
    perf_rd = (1 + df['strat_random'].dropna()).prod() - 1
    perf_rsi = (1 + df['strat_rsi'].dropna()).prod() - 1
    
    print(f"Performance Buy & Hold : {perf_bh:.2%}")
    print(f"Performance Aléatoire : {perf_rd:.2%}")
    print(f"Performance Stratégie RSI : {perf_rsi:.2%}")
    
    plt.figure(figsize=(12, 6))
    (1 + df[['strat_buy_hold', 'strat_random', 'strat_rsi']].fillna(0)).cumprod().plot(ax=plt.gca())
    plt.title("Comparaison des Stratégies Baseline")
    plt.savefig(f"{out_path}/baseline_performance.png")
    plt.close()
    
    print("T06 Terminée : Comparaison sauvegardée.")
    return df

def task_07_machine_learning(df, out_path):
    print("\n--- DÉBUT T07 : MACHINE LEARNING ---")
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
                'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
                'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14']
    
    df_ml = df.dropna(subset=['target'] + features)
    X = df_ml[features]
    y = df_ml['target']
    
    split = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Précision du modèle (Accuracy) : {accuracy_score(y_test, y_pred):.2%}")
    
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title("Importance des Features")
    plt.savefig(f"{out_path}/feature_importance.png")
    plt.close()

    print("T07 Terminée : Modèle ML entraîné.")
    return df

def task_08_rl_env_setup(df):
    print("\n--- DÉBUT T08 : INITIALISATION ENVIRONNEMENT RL ---")
    env = TradingEnv(df)
    obs, _ = env.reset()
    print(f"T08 Terminée : Environnement Gym créé. (Shape obs: {obs.shape})")
    return env

def task_09_train_rl(env):
    print("\n--- DÉBUT T09 : ENTRAÎNEMENT DE L'AGENT RL (PPO) ---")
    
    # On définit le modèle
    # MlpPolicy : Réseau de neurones classique (parfait pour des données tabulaires)
    # verbose=1 : Pour voir l'évolution de la récompense dans la console
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    
    print("Lancement de l'apprentissage (10 000 timesteps)...")
    model.learn(total_timesteps=10000)
    
    # Sauvegarde dans le dossier output (pour qu'il arrive sur ton PC)
    model_path = "output/ppo_trading_model"
    model.save(model_path)
    
    print(f"T09 Terminée : Modèle sauvegardé sous {model_path}.zip")
    return model

def task_10_backtest(df, model, out_path):
    print("\n--- DÉBUT T10 : BACKTEST DU MODÈLE RL ---")
    # On crée un environnement de test avec les mêmes données
    env = TradingEnv(df)
    obs, _ = env.reset()
    
    rewards = []
    actions = []
    done = False
    
    print("Simulation des trades en cours...")
    while not done:
        # L'IA prédit l'action sans exploration (deterministic=True)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        rewards.append(reward)
        actions.append(action)
    
    # Analyse des résultats
    df_res = pd.DataFrame({'reward': rewards, 'action': actions})
    
    # 0 = Cash, 1 = Long, 2 = Short. On calcule le rendement cumulé.
    # Note : Le reward est déjà le next_return (ou -next_return)
    df_res['cum_return'] = (1 + df_res['reward']).cumprod()
    
    # Graphique de performance
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['cum_return'], label='Agent RL (PPO)', color='blue')
    plt.axhline(y=1, color='red', linestyle='--', label='Break-even')
    plt.title("Backtest : Performance Cumulée de l'Agent IA")
    plt.xlabel("Nombre de bougies (15 min)")
    plt.ylabel("Multiplicateur de Capital")
    plt.legend()
    plt.savefig(f"{out_path}/backtest_rl.png")
    plt.close()
    
    # Statistiques rapides
    final_perf = (df_res['cum_return'].iloc[-1] - 1) * 100
    print(f"Nombre de trades simulés : {len(df_res)}")
    print(f"Répartition des actions : 0(Rien):{actions.count(0)}, 1(Achat):{actions.count(1)}, 2(Vente):{actions.count(2)}")
    print(f"T10 Terminée : Performance totale : {final_perf:.2f}%")

   
    
 
if __name__ == "__main__":
    # Définition des dossiers
    DATA_DIR = 'data/'
    OUT_DIR = 'output/'
    os.makedirs(OUT_DIR, exist_ok=True) # Crée le dossier output s'il n'existe pas

    df = task_01_import_m1(DATA_DIR)
    df = task_02_aggregate_m1_to_m15(df)
    df = task_03_clean_m15(df)
    df = task_04_exploratory_analysis(df, OUT_DIR)
    df = task_05_feature_engineering(df)
    df = task_06_baseline_strategies(df, OUT_DIR)
    df = task_07_machine_learning(df, OUT_DIR)
    
    # Task 08
    env = task_08_rl_env_setup(df)

    # Task 09 : Training
    model_rl = task_09_train_rl(env)
    
    # Task 10 : Backtest
    task_10_backtest(df, model_rl, OUT_DIR)
    
    # Sauvegarde finale
    df.to_csv(f'{OUT_DIR}/gbpusd_final_features.csv', index=False)
    print(f"\n🚀 Pipeline terminé ! Vérifie le graphique backtest_rl.png dans /{OUT_DIR}")