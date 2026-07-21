import pandas as pd
import numpy as np
import pandas_ta as ta  # pip install pandas-ta (gratuito)

def add_custom_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona indicadores técnicos robustos"""
    if len(df) < 50:
        return df
    
    # Indicadores básicos pandas-ta
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['ATRr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ADX_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)
    
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    
    # Volume indicators
    df['OBV'] = ta.obv(df['close'], df['volume'])
    
    # Custom HMM regime (se disponível no treinador)
    # ... (pode ser expandido)
    
    return df
