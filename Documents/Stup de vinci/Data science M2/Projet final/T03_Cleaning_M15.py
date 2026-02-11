import pandas as pd
import numpy as np

def clean_m15_data(input_path, output_path):
    print("--- Démarrage T03 : Nettoyage M15 ---")
    
    # Chargement du dataset agrégé
    df = pd.read_parquet(input_path)
    initial_len = len(df)
    
    # 1. Suppression des bougies incomplètes (NaN) [cite: 42]
    # Le resample peut créer des lignes vides pour les week-ends
    df = df.dropna()
    
    # 2. Contrôle des prix négatifs ou nuls [cite: 43]
    # Sur le Forex, un prix <= 0 est une erreur de donnée critique
    invalid_prices = df[(df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)]
    if not invalid_prices.empty:
        print(f"Alerte : {len(invalid_prices)} bougies avec prix invalides supprimées.")
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    
    # 3. Détection des Gaps anormaux [cite: 44]
    # On calcule l'écart entre le Close précédent et l'Open actuel
    df['gap'] = (df['open'] - df['close'].shift(1)).abs()
    
    # On définit un gap "anormal" comme étant 5x supérieur à la moyenne des gaps
    gap_threshold = df['gap'].mean() + 5 * df['gap'].std()
    gaps_anormaux = df[df['gap'] > gap_threshold]
    
    print(f"Rapport de nettoyage :")
    print(f"- Bougies initiales : {initial_len}")
    print(f"- Bougies après dropna/prix : {len(df)}")
    print(f"- Gaps importants détectés : {len(gaps_anormaux)}")
    
    # On retire la colonne technique 'gap' avant de sauvegarder
    df = df.drop(columns=['gap'])
    
    # Sauvegarde du dataset prêt pour l'analyse et les features
    df.to_parquet(output_path)
    print(f"Fichier nettoyé sauvegardé : {output_path}")
    return df

# Exécution
if __name__ == "__main__":
    clean_m15_data("C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/GBPUSD_M15_CLEAN.parquet", "C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/GBPUSD_M15_FINAL.parquet")