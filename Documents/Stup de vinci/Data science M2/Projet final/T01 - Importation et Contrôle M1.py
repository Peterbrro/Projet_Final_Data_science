import pandas as pd
import glob
import os

DATA_PATH = "C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/"
OUTPUT_FILE = "C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/GBPUSD_M1_CLEAN.parquet"

def charger_et_nettoyer_m1():
    print("--- Démarrage Tâche T01 : Import M1 ---")
    
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    all_data = []
    
    # Noms de colonnes standards pour ce format (Date, Time, Open, High, Low, Close, Vol)
    col_names = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
    
    for file in csv_files:
        print(f"Chargement de : {file}")
        
        # On lit sans header et on donne les noms nous-mêmes
        df = pd.read_csv(file, header=None, names=col_names, sep=',')
        
        # 2. Fusion Date + Time -> Timestamp [cite: 32]
        # On remplace les points par des tirets pour que pandas comprenne mieux (2022.01.02 -> 2022-01-02)
        df['datetime'] = pd.to_datetime(df['date'].str.replace('.', '-') + ' ' + df['time'])
        
        # On ne garde que l'essentiel
        df = df[['datetime', 'open', 'high', 'low', 'close']].copy()
        all_data.append(df)

    if not all_data:
        print("Erreur : Aucune donnée chargée.")
        return None

    full_df = pd.concat(all_data)
    
    # 3. Tri Chronologique [cite: 34]
    full_df = full_df.sort_values('datetime').set_index('datetime')
    
    # Suppression des doublons
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    # 4. Vérification Régularité 1 minute [cite: 33]
    # Attention : Le Forex ferme le week-end ! 
    # On va juste compter les trous sans créer de lignes vides pour l'instant.
    full_range = pd.date_range(start=full_df.index.min(), end=full_df.index.max(), freq='1min')
    # On filtre pour ne garder que les jours de semaine (Lundi=0, Vendredi=4) pour un calcul plus juste
    weekdays_range = full_range[full_range.dayofweek < 5]
    missing_minutes = weekdays_range.difference(full_df.index)
    
    print(f"\n--- Rapport de Qualité ---")
    print(f"Période : {full_df.index.min()} à {full_df.index.max()}")
    print(f"Total bougies M1 : {len(full_df)}")
    print(f"Trous détectés (hors WE approximatif) : {len(missing_minutes)}")
    
    # Vérification Split Temporel Obligatoire [cite: 23, 25, 26, 27]
    for year in [2022, 2023, 2024]:
        count = len(full_df[full_df.index.year == year])
        print(f"Année {year} : {count} bougies")

    full_df.to_parquet(OUTPUT_FILE)
    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")
    return full_df

df_m1 = charger_et_nettoyer_m1()