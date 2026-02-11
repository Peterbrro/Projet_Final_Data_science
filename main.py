import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

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
    
    # 1. Distribution des rendements
    df['returns'] = df['close'].pct_change()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['returns'].dropna(), bins=100, kde=True)
    plt.title("Distribution des Rendements GBP/USD")
    plt.savefig("data/distribution_rendements.png")
    
    # 2. Analyse horaire (Volatilité)
    df['hour'] = df['timestamp'].dt.hour
    hourly_vol = df.groupby('hour')['returns'].std()
    plt.figure(figsize=(10, 6))
    hourly_vol.plot(kind='bar')
    plt.title("Volatilité par Heure (Saisonnalité)")
    plt.savefig("data/volatilite_horaire.png")

    # 3. Test ADF (Stationnarité)
    print("Calcul du Test ADF...")
    adf_test = adfuller(df['returns'].dropna())
    print(f"Statistique ADF : {adf_test[0]:.4f}")
    print(f"p-value : {adf_test[1]:.4e}") # On attend < 0.05
    
    # 4. Autocorrélation (ACF)
    plt.figure(figsize=(10, 6))
    plot_acf(df['returns'].dropna(), lags=40)
    plt.savefig("data/autocorrelation_acf.png")
    
    print("T04 Terminée : Graphiques sauvegardés dans /data.")
    return df

if __name__ == "__main__":
    df = task_01_import_m1('data/')
    df = task_02_aggregate_m1_to_m15(df)
    df = task_03_clean_m15(df)
    df = task_04_exploratory_analysis(df)
    
    # Sauvegarde finale pour la T05
    df.to_csv('data/gbpusd_m15_eda.csv', index=False)