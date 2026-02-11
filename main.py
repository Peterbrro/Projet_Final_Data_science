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

if __name__ == "__main__":
    df_m1 = task_01_import_m1('data/')