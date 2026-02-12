import pandas as pd
import glob
import os

def task_01_import_m1(data_path):
    print("\n--- T01 : IMPORTATION ---")
    csv_files = sorted(glob.glob(os.path.join(data_path, "DAT_MT_*.csv")))
    if not csv_files:
        print(f"⚠️ Erreur : Aucun fichier trouvé dans {data_path}")
        return None
    
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
    li = []
    for f in csv_files:
        df = pd.read_csv(f, names=cols, sep=',', index_col=False)
        li.append(df)
    
    full_df = pd.concat(li, axis=0, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['date'] + ' ' + full_df['time'], format='%Y.%m.%d %H:%M')
    
    # Nettoyage doublons et tri
    full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'vol']]
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    print(f"T01 Terminée : {len(full_df)} lignes M1 importées.")
    return full_df

def task_02_03_aggregate_and_clean(df_m1):
    print("\n--- T02/T03 : AGRÉGATION M15 & NETTOYAGE ---")
    # T02 : Resampling
    df_m15 = df_m1.set_index('timestamp').resample('15min', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).dropna().reset_index()
    
    # T03 : Nettoyage prix aberrants
    df_clean = df_m15[(df_m15['open'] > 0) & (df_m15['high'] >= df_m15['low'])].copy()
    print(f"T02/T03 Terminées : {len(df_clean)} bougies M15 prêtes.")
    return df_clean