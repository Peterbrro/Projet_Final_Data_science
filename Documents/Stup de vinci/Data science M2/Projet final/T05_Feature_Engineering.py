import pandas as pd
import numpy as np
import pandas_ta as ta

# 1. Chargement des données M15 nettoyées
df = pd.read_parquet("data/GBPUSD_M15_FINAL.parquet")

def add_features(df):
    print("--- T05 : Calcul du Feature Pack V2 ---")
    
    # --- BLOC COURT TERME ---
    # Rendements passés
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    
    # Indicateurs de tendance et momentum
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_diff'] = df['ema_20'] - df['ema_50']
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # Volatilité et Chandeliers
    df['rolling_std_20'] = df['close'].rolling(20).std()
    df['range_15m'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # --- BLOC CONTEXTE & RÉGIME ---
    # Tendance long terme
    df['ema_200'] = ta.ema(df['close'], length=200)
    df['dist_ema200'] = df['close'] - df['ema_200']
    # Pente de l'EMA50 (Slope)
    df['slope_ema50'] = df['ema_50'].diff(3) 
    
    # Régime de volatilité
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rolling_std_100'] = df['close'].rolling(100).std()
    df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_100']
    
    # Force directionnelle
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx_14'] = adx['ADX_14']
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    
    # Suppression des lignes avec des NaNs créés par les indicateurs (Warm-up)
    df = df.dropna()
    
    return df

# Exécution et sauvegarde
df_features = add_features(df)
df_features.to_parquet("data/GBPUSD_M15_FINAL.parquet")

print(f"Feature Engineering terminé. Nombre de colonnes : {len(df_features.columns)}")
print(f"Fichier sauvegardé : data/GBPUSD_M15_FINAL.parquet")