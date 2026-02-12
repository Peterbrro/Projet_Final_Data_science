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

def task_01_import_m1(data_path):
    print("--- DÉBUT T01 : IMPORTATION ET VÉRIFICATION ---")
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
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
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum'
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

def task_04_exploratory_analysis(df):
    print("\n--- DÉBUT T04 : ANALYSE EXPLORATOIRE ---")
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['returns'].dropna(), bins=100, kde=True)
    plt.title("Distribution des Rendements GBP/USD")
    plt.savefig("data/distribution_rendements.png")
    
    df['hour'] = df['timestamp'].dt.hour
    hourly_vol = df.groupby('hour')['returns'].std()
    plt.figure(figsize=(10, 6))
    hourly_vol.plot(kind='bar')
    plt.title("Volatilité par Heure (Saisonnalité)")
    plt.savefig("data/volatilite_horaire.png")

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

    # --- 6.1 BLOC COURT TERME ---
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

    # --- 6.2 BLOC CONTEXTE & RÉGIME ---
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

def task_06_baseline_strategies(df):
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
    plt.savefig("data/baseline_performance.png")
    plt.close()
    
    print("T06 Terminée : Comparaison sauvegardée.")
    return df

def task_07_machine_learning(df):
    print("\n--- DÉBUT T07 : MACHINE LEARNING ---")
    df = df.copy()
    
    # Target : 1 si le prix monte à la bougie suivante, sinon 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Sélection des features (uniquement celles calculées en T05)
    # Note : ADX_14, DMP_14, DMN_14 sont les noms générés par pandas_ta
    features = ['return_1', 'return_4', 'ema_diff', 'rsi_14', 'rolling_std_20', 
                'range_15m', 'body', 'upper_wick', 'lower_wick', 'distance_to_ema200',
                'slope_ema50', 'atr_14', 'volatility_ratio', 'ADX_14', 'DMP_14', 'DMN_14']
    
    df_ml = df.dropna(subset=['target'] + features)
    X = df_ml[features]
    y = df_ml['target']
    
    # Split temporel (80% train, 20% test)
    split = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Entraînement sur {len(X_train)} lignes, Test sur {len(X_test)} lignes.")
    
    # Modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle (Accuracy) : {acc:.2%}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    # Importance des features
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title("Importance des Features")
    plt.savefig("data/feature_importance.png")
    plt.close()

    print("T07 Terminée : Modèle ML entraîné.")
    return df

if __name__ == "__main__":
    df = task_01_import_m1('data/')
    df = task_02_aggregate_m1_to_m15(df)
    df = task_03_clean_m15(df)
    df = task_04_exploratory_analysis(df)
    df = task_05_feature_engineering(df)
    df = task_06_baseline_strategies(df)
    df = task_07_machine_learning(df)
    
    df.to_csv('data/gbpusd_final_ml.csv', index=False)
    print("\n Toutes les étapes terminées. Prêt pour la Phase 8 (RL) !")