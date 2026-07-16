import pandas as pd
import numpy as np

def add_custom_ta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motor interno de Análise Técnica Vetorizada.
    Substitui completamente a dependência externa do pandas-ta.
    """
    # 1. EMA 20
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # 2. Bollinger Bands (20, 2)
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['BBL_20_2.0'] = sma20 - (2 * std20)
    df['BBU_20_2.0'] = sma20 + (2 * std20)

    # 3. RSI 14 (Wilder's Smoothing)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/14, adjust=False).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False).mean()
    # Proteção 1e-9 contra divisão por zero em momentos sem volatilidade
    rs = ema_up / (ema_down + 1e-9)
    df['RSI_14'] = np.where(ema_down == 0, 100, 100 - (100 / (1 + rs)))

    # 4. ATR 14 (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATRr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # 5. ADX 14 (Average Directional Index)
    up_m = df['high'] - df['high'].shift()
    down_m = df['low'].shift() - df['low']
    plus_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)

    tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()

    # Proteção 1e-9 nos denominadores do ADX
    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-9)
    df['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    return df
