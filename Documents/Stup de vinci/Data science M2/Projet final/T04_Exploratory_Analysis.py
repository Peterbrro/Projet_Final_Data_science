import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# 1. Chargement des données nettoyées
df = pd.read_parquet("C:/Users/peter/Documents/Stup de vinci/Data science M2/Projet final/data/GBPUSD_M15_FINAL.parquet")

# Calcul des log-rendements (indispensable pour l'analyse statistique)
df['returns'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna()

print("--- T04 : Analyse Exploratoire ---")

# 2. Test de Stationnarité (ADF Test)
# Obligatoire pour savoir si on peut modéliser la série
def check_stationarity(series):
    result = adfuller(series)
    print(f'Statistique ADF : {result[0]:.4f}')
    print(f'p-value : {result[1]:.4e}')
    if result[1] <= 0.05:
        print("=> La série est stationnaire (p <= 0.05)")
    else:
        print("=> La série n'est pas stationnaire")

print("\nTest ADF sur les prix (Close) :")
check_stationarity(df['close'])
print("\nTest ADF sur les rendements :")
check_stationarity(df['returns'])

# 3. Visualisations
plt.figure(figsize=(15, 10))

# Sous-graphique 1 : Distribution des rendements
plt.subplot(2, 2, 1)
sns.histplot(df['returns'], kde=True, bins=100)
plt.title("Distribution des Rendements (Log-Returns)")

# Sous-graphique 2 : Volatilité (Rolling Std)
plt.subplot(2, 2, 2)
df['returns'].rolling(window=100).std().plot()
plt.title("Volatilité (Écart-type glissant 100)")

# Sous-graphique 3 : Analyse Horaire (Effet de session)
plt.subplot(2, 2, 3)
df['hour'] = df.index.hour
df.groupby('hour')['returns'].std().plot(kind='bar')
plt.title("Volatilité moyenne par heure de la journée")
plt.xlabel("Heure")

# Sous-graphique 4 : Autocorrélation (ACF)
plt.subplot(2, 2, 4)
plot_acf(df['returns'], lags=40, ax=plt.gca())
plt.title("Autocorrélation des rendements")

plt.tight_layout()
plt.show()

# Sauvegarde des résultats pour le rapport qualité
print("\nAnalyse terminée. Vérifie les graphiques pour détecter des anomalies.")