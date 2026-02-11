import pandas as pd
import glob
import os
import re

def task_01_import_m1(data_path):
    print("--- DÉBUT T01 : IMPORTATION ET VÉRIFICATION ---")
    
    # 1. Chargement des CSV (Données de prix)
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
    
    li = []
    for f in csv_files:
        df = pd.read_csv(f, names=cols, sep=',', index_col=False)
        li.append(df)
    
    full_df = pd.concat(li, axis=0, ignore_index=True)
    
    # 2. Fusion Date + Time -> Timestamp
    full_df['timestamp'] = pd.to_datetime(full_df['date'] + ' ' + full_df['time'], format='%Y.%m.%d %H:%M')
    full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'vol']]
    
    # 3. Tri et suppression des doublons
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # 4. Détection automatique des Gaps (Code)
    time_diff = full_df['timestamp'].diff().dt.total_seconds()
    auto_gaps = time_diff[time_diff > 60]
    
    # 5. Lecture des fichiers TXT (Rapports HistData) pour comparaison
    txt_files = glob.glob(os.path.join(data_path, "*.txt"))
    official_gaps_count = 0
    for f in txt_files:
        with open(f, 'r') as file:
            official_gaps_count += len(re.findall(r"Gap of", file.read()))

    print(f"Importation réussie : {len(full_df)} lignes.")
    print(f"Période : {full_df['timestamp'].min()} au {full_df['timestamp'].max()}")
    print(f"Gaps détectés par le code : {len(auto_gaps)}")
    print(f"Gaps listés par HistData (.txt) : {official_gaps_count}")
    
    return full_df

def task_02_aggregate_m1_to_m15(df_m1):
    print("\n--- DÉBUT T02 : AGRÉGATION M1 -> M15 ---")
    
    # Copie pour éviter de modifier l'original et mise en index pour le resample
    df_resample = df_m1.set_index('timestamp').copy()
    
    # Agrégation selon les règles strictes du sujet (Section 3.2)
    # label='left' : le groupe 17:00 contient les données de 17:00 à 17:14
    df_m15 = df_resample.resample('15min', label='left').agg({
        'open': 'first',   # open_15m : open 1ère minute
        'high': 'max',     # high_15m : max(high) sur 15 minutes
        'low': 'min',      # low_15m : min(low) sur 15 minutes
        'close': 'last',   # close_15m : close dernière minute
        'vol': 'sum'       # Volume total sur la période
    })
    
    # Suppression des bougies sans données (week-ends / marchés fermés)
    df_m15 = df_m15.dropna(subset=['open']).reset_index()
    
    print(f"Agrégation terminée : {len(df_m15)} bougies M15 créées.")
    
    return df_m15

if __name__ == "__main__":
    # Exécution Phase 1
    df_m1 = task_01_import_m1('data/')
    
    # Exécution Phase 2
    if df_m1 is not None:
        df_m15 = task_02_aggregate_m1_to_m15(df_m1)
        
        # Affichage des premières lignes pour vérification
        print("\n--- APERÇU DONNÉES M15 ---")
        print(df_m15.head())