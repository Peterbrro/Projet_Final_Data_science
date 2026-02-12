import pandas_ta as ta
import pandas as pd

def calculate_features(df):
    df = df.copy()
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    df['next_return'] = df['close'].pct_change().shift(-1)
    df['ema_20'] = ta.ema(df['close'], 20)
    df['ema_50'] = ta.ema(df['close'], 50)
    df['ema_200'] = ta.ema(df['close'], 200)
    df['ema_diff'] = df['ema_20'] - df['ema_50']
    df['distance_to_ema200'] = df['close'] - df['ema_200']
    df['slope_ema50'] = df['ema_50'].diff(5)
    df['rsi_14'] = ta.rsi(df['close'], 14)
    df['rolling_std_20'] = df['close'].rolling(20).std()
    df['rolling_std_100'] = df['close'].rolling(100).std()
    df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_100']
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], 14)
    df['range_15m'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    adx = ta.adx(df['high'], df['low'], df['close'], 14)
    df = pd.concat([df, adx], axis=1)
    return df.dropna().reset_index(drop=True)