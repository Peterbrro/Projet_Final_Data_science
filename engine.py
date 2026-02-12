import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from trading_env import TradingEnv

def task_04_eda(df, out_path):
    print("\n--- T04 : ANALYSE EXPLORATOIRE (EDA) ---")
    df = df.copy()
    df['returns_eda'] = df['close'].pct_change()
    
    # Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['returns_eda'].dropna(), bins=100, kde=True)
    plt.title("Distribution des Rendements GBP/USD")
    plt.savefig(f"{out_path}/T04_distribution.png")
    
    # Test ADF (Stationnarité)
    print("Calcul du Test ADF...")
    adf_test = adfuller(df['returns_eda'].dropna())
    print(f"Statistique ADF : {adf_test[0]:.4f} | p-value : {adf_test[1]:.4e}")
    plt.close()
    return df

def task_06_baselines(df_test, out_path):
    print("\n--- T06 : STRATÉGIES BASELINE (SUR 2024) ---")
    df = df_test.copy()
    # On utilise next_return calculé dans features.py
    df['strat_buy_hold'] = df['next_return']
    df['strat_random'] = np.random.choice([1, -1], size=len(df)) * df['next_return']
    
    df['signal_rsi'] = 0
    df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1
    df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1
    df['strat_rsi'] = df['signal_rsi'] * df['next_return']
    
    # Graphique
    plt.figure(figsize=(12, 6))
    (1 + df[['strat_buy_hold', 'strat_random', 'strat_rsi']].fillna(0)).cumprod().plot(ax=plt.gca())
    plt.title("Comparaison Baselines vs Buy & Hold (2024)")
    plt.savefig(f"{out_path}/T06_baselines_2024.png")
    plt.close()
    print("T06 Terminée : Graphique des baselines généré.")

def task_07_ml(df_train, df_test, out_path):
    print("\n--- T07 : MACHINE LEARNING (RF) ---")
    features = ['return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
                'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
                'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14']
    
    # Target : Est-ce que ça monte à la prochaine bougie ?
    y_train = (df_train['next_return'] > 0).astype(int)
    y_test = (df_test['next_return'] > 0).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(df_train[features], y_train)
    
    acc = accuracy_score(y_test, model.predict(df_test[features]))
    print(f"Précision ML sur 2024 : {acc:.2%}")
    return model

def task_09_eval_robuste(df_2024, model_rl, out_path):
    print("\n--- T09 : ÉVALUATION ROBUSTE RL (2024) ---")
    env = TradingEnv(df_2024)
    obs, _ = env.reset()
    rewards, actions = [], []
    done = False
    
    while not done:
        action, _ = model_rl.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
    
    res = pd.DataFrame({'reward': rewards, 'action': actions})
    res['cum_return'] = (1 + res['reward']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(res['cum_return'], label='Agent RL')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title("Performance Agent RL (Année Test 2024)")
    plt.savefig(f"{out_path}/T09_performance_RL_2024.png")
    plt.close()
    
    final_pnl = (res['cum_return'].iloc[-1] - 1) * 100
    print(f"Performance RL 2024 : {final_pnl:.2f}%")