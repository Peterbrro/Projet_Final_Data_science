import pandas as pd

# On charge le fichier propre de l'étape T01
df_m1 = pd.read_parquet("C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/data/GBPUSD_M1_CLEAN.parquet")

print("Début de l'agrégation M15...")

# Règles d'agrégation OHLC
logic = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
}

# Resample en 15 minutes
df_m15 = df_m1.resample('15min', label='left', closed='left').agg(logic)

# Suppression des bougies vides (ex: week-ends)
df_m15 = df_m15.dropna()

# Sauvegarde pour la suite (T03/T04)
df_m15.to_parquet("data/GBPUSD_M15_CLEAN.parquet")

print(f"Agrégation terminée. {len(df_m15)} bougies M15 générées.")