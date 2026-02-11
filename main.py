import pandas as pd
import glob
import os
import re

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
    
    time_diff = full_df['timestamp'].diff().dt.total_seconds()
    auto_gaps = time_diff[time_diff > 60]
    txt_files = glob.glob(os.path.join(data_path, "*.txt"))
    official_gaps_count = 0
    for f in txt_files:
        with open(f, 'r') as file:
            official_gaps_count += len(re.findall(r"Gap of", file.read()))

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
    initial_count = len(df_clean)
    
    # 1. Contrôle des prix négatifs ou nuls
    df_clean = df_clean[(df_clean['open'] > 0) & (df_clean['high'] > 0) & 
                        (df_clean['low'] > 0) & (df_clean['close'] > 0)]
    
    # 2. Détection gaps anormaux (Écart temporel > 15 min hors weekend)
    # On calcule l'écart entre deux bougies consécutives
    time_gap = df_clean['timestamp'].diff().dt.total_seconds()
    # Un gap anormal est un saut > 15 min (900s)
    gaps_anormaux = time_gap[time_gap > 900]
    
    # 3. Suppression bougies incomplètes (Cohérence OHLC)
    # Le High doit être >= au Low et aux autres prix
    df_clean = df_clean[(df_clean['high'] >= df_clean['low']) & 
                        (df_clean['high'] >= df_clean['open']) & 
                        (df_clean['low'] <= df_clean['close'])]

    print(f"T03 Terminée.")
    print(f"Lignes supprimées : {initial_count - len(df_clean)}")
    print(f"Nombre de gaps temporels (>15min) détectés en M15 : {len(gaps_anormaux)}")
    
    return df_clean

if __name__ == "__main__":
    df_m1 = task_01_import_m1('data/')
    if df_m1 is not None:
        df_m15 = task_02_aggregate_m1_to_m15(df_m1)
        df_final = task_03_clean_m15(df_m15)
        
        # Sauvegarde du dataset prêt pour l'analyse
        df_final.to_csv('data/gbpusd_m15_clean.csv', index=False)
        print(f"Dataset final prêt : {len(df_final)} bougies.")