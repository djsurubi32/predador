import pandas as pd
import numpy as np

# Quantidade de barras a descartar na preparação final do modelo
# Garante a convergência matemática e elimina falsas "features" (warmup)
WARMUP_BARS = 100

def add_custom_ta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motor interno de Análise Técnica Vetorizada de Alta Performance.
    Substitui completamente a dependência externa do pandas-ta[cite: 3].
    """
    # 1. Validação Estrutural e de Contrato
    if len(df) < WARMUP_BARS:
        raise ValueError(f"Série temporal truncada. Mínimo exigido: {WARMUP_BARS} barras para convergência.")

    if 'timestamp' in df.columns and not df['timestamp'].is_monotonic_increasing:
        raise ValueError("Falha de ordenação cronológica detectada no DataFrame.")

    # Proteção do chamador (Prevenção de mutação in-place)
    out = df.copy()

    # 2. EMA 20
    out['EMA_20'] = out['close'].ewm(span=20, adjust=False).mean()

    # 3. Bollinger Bands (20, 2) com desvio populacional real (ddof=0)
    sma20 = out['close'].rolling(window=20).mean()
    std20 = out['close'].rolling(window=20).std(ddof=0)
    out['BBL_20_2.0'] = sma20 - (2 * std20)
    out['BBU_20_2.0'] = sma20 + (2 * std20)

    # 4. RSI 14 (Lógica algébrica sem divisão por epsilon)
    delta = out['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(alpha=1/14, adjust=False).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False).mean()

    soma_rsi = ema_up + ema_down
    # Em caso de paralisia de mercado (soma == 0), reverte ao centro natural de força (50.0)
    out['RSI_14'] = np.where(soma_rsi > 0, 100.0 * ema_up / soma_rsi, 50.0)

    # 5. ATR 14 (Otimização vetorizada usando matrizes C-contíguas)
    tr1 = out['high'] - out['low']
    tr2 = (out['high'] - out['close'].shift()).abs()
    tr3 = (out['low'] - out['close'].shift()).abs()

    tr_arr = np.maximum(tr1.to_numpy(), np.maximum(tr2.to_numpy(), tr3.to_numpy()))
    tr = pd.Series(tr_arr, index=out.index)
    out['ATRr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # 6. ADX 14 (Cancelamento analítico do True Range e do epsilon)
    up_m = out['high'] - out['high'].shift()
    down_m = out['low'].shift() - out['low']

    plus_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)

    plus_dm_smooth = pd.Series(plus_dm, index=out.index).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=out.index).ewm(alpha=1/14, adjust=False).mean()

    dm_sum = plus_dm_smooth + minus_dm_smooth
    dx = np.where(dm_sum > 0, 100.0 * (plus_dm_smooth - minus_dm_smooth).abs() / dm_sum, 0.0)

    dx_series = pd.Series(dx, index=out.index)
    out['ADX_14'] = dx_series.ewm(alpha=1/14, adjust=False).mean()

    return out
