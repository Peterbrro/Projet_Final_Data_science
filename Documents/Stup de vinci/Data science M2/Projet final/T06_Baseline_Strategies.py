import pandas as pd
import numpy as np

# Chargement du dataset enrichi à la Phase 5
df = pd.read_parquet("data/GBPUSD_M15_FEATURES.parquet")

# On s'assure d'avoir les rendements pour calculer les performances
df['returns'] = df['close'].pct_change()
df = df.dropna()

print("--- T06 : Calcul des Baselines ---")

# 1. Stratégie Buy & Hold (Achat au début, vente à la fin)
# On simule un investissement qui suit simplement le cours
df['strat_buy_hold'] = (1 + df['returns']).cumprod()

# 2. Stratégie Aléatoire (Random)
# Pile ou face à chaque bougie (1 pour Buy, -1 pour Sell)
np.random.seed(42)
df['random_signal'] = np.random.choice([-1, 1], size=len(df))
df['strat_random'] = (1 + df['random_signal'].shift(1) * df['returns']).cumprod()

# 3. Stratégie à Règles Fixes (Exemple : RSI classique)
# Achat si RSI < 30 (survendu), Vente si RSI > 70 (suracheté)
df['rsi_signal'] = 0
df.loc[df['rsi_14'] < 30, 'rsi_signal'] = 1 
df.loc[df['rsi_14'] > 70, 'rsi_signal'] = -1
df['strat_rsi'] = (1 + df['rsi_signal'].shift(1) * df['returns']).cumprod()

# Affichage des résultats pour le rapport
print(f"Performance finale cumulée :")
print(f"- Buy & Hold : {df['strat_buy_hold'].iloc[-1]:.4f}")
print(f"- Random     : {df['strat_random'].iloc[-1]:.4f}")
print(f"- RSI Fixe   : {df['strat_rsi'].iloc[-1]:.4f}")

# Sauvegarde pour la comparaison finale (Phase 10)
df.to_parquet("data/GBPUSD_M15_BASELINES.parquet")
print("\nFichier sauvegardé dans data/GBPUSD_M15_BASELINES.parquet")