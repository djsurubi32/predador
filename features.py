"""
features.py — Pipeline unico de features do Predador v3.2.

REGRA DE OURO:
    Treino e inferencia chamam adicionar_features() com o MESMO contrato
    de dados brutos e recebem de volta o MESMO conjunto de colunas.

Contrato de entrada (df):
    timestamp, open, high, low, close, volume,
    open_interest, funding_rate        (preencher com 0.0 se ausente)

Contrato de entrada (btc_df, opcional):
    timestamp, close                    (benchmark BTC/USDT)

Saida:
    df original + colunas de FEATURE_NAMES, com as primeiras WARMUP_BARS
    linhas descartadas (warmup matematico) e sem NaN/inf.
    A ULTIMA linha do df de saida e sempre a vela mais recente.

Uso no treino:      df_feat = adicionar_features(df_bruto, btc_df)
Uso na inferencia:  df_feat = adicionar_features(df_ohlcv_recente, btc_df)
"""

import numpy as np
import pandas as pd

# Barras descartadas no inicio para convergencia das janelas rolantes
# e do resample multi-timeframe (4h precisa de ~100 barras de 5m).
WARMUP_BARS = 150

# Quantidade minima de candles 5m que o analisador DEVE buscar.
OHLCV_LIMIT_INFERENCIA = 500  # ~41 horas: sobra margem sobre WARMUP_BARS

RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume",
               "open_interest", "funding_rate"]


# ---------------------------------------------------------------------
# Indicadores internos vetorizados (sem dependencia externa)
# ---------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    soma = up + down
    return pd.Series(np.where(soma > 0, 100.0 * up / soma, 50.0), index=close.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    atr = _atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    ll = low.rolling(period, min_periods=period).min()
    hh = high.rolling(period, min_periods=period).max()
    return 100 * (close - ll) / (hh - ll + 1e-12)


def _resample_mtf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Fecha em timeframe maior usando APENAS dados passados.
    Aplica .shift(1) para garantir zero lookahead bias: a barra atual de 5m
    enxerga apenas velas MTF que ja fecharam completamente.
    """
    idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    s = pd.Series(df["close"].values, index=idx)
    mtf_close = s.resample(rule).last().shift(1).ffill()
    out = pd.DataFrame(index=mtf_close.index)
    out["ema20"] = mtf_close.ewm(span=20, adjust=False, min_periods=5).mean()
    out["rsi"] = _rsi(mtf_close).reindex(out.index).ffill()
    out["macdh"] = _macd_hist(mtf_close).reindex(out.index).ffill()
    return out.ffill()


# ---------------------------------------------------------------------
# FEATURE NAMES — contrato publico entre treino e inferencia
# ---------------------------------------------------------------------
FEATURE_NAMES = [
    # --- Retorno e volatilidade ---
    "log_return", "ret_3", "ret_6", "volatility_20", "natr", "range_pct",
    # --- Acao do preco ---
    "body_pct", "close_pos", "upper_wick", "lower_wick",
    # --- Tendencia ---
    "ema_fast_ratio", "ema_trend_ratio", "price_vs_ema21", "ema21_slope", "adx",
    # --- Momento ---
    "rsi", "rsi_slope", "stoch_k", "stoch_d", "macd_hist", "macd_hist_slope", "roc_6",
    # --- Volatilidade estrutural ---
    "bb_pos", "bb_width", "squeeze",
    # --- Volume / fluxo ---
    "vol_z", "vol_ratio", "cvd_slope", "obv_slope",
    # --- Derivativos (OI + funding) ---
    "oi_change", "oi_z", "oi_trend", "oi_price_div",
    "funding", "funding_ema", "funding_delta",
    # --- Multi-timeframe ---
    "mtf1h_dist", "mtf1h_rsi", "mtf1h_macdh",
    "mtf4h_dist", "mtf4h_rsi", "mtf4h_macdh",
    # --- Contexto BTC ---
    "btc_corr", "rel_strength_24",
    # --- Relogio do mercado (funding em horario fixo, sessoes) ---
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Features que o HMM usa (calculadas aqui, consumidas no treinador/rotulador)
HMM_INPUTS = ["log_return", "volatility_20"]


# ---------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------
def adicionar_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Retorna df com FEATURE_NAMES prontas, warmup descartado, sem NaN/inf.
    Funcao deterministica: mesma entrada -> mesma saida, em treino e ao vivo.
    """
    out = df.copy()

    # Garante o contrato de colunas brutas
    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.ffill(inplace=True)
    out.fillna({"open_interest": 0.0, "funding_rate": 0.0}, inplace=True)

    eps = 1e-12
    c, h, l, o, v = out["close"], out["high"], out["low"], out["open"], out["volume"]
    oi, fr = out["open_interest"], out["funding_rate"]

    # ================= RETORNO E VOLATILIDADE =================
    out["log_return"] = np.log(c / c.shift(1).replace(0, np.nan))
    out["ret_3"] = c.pct_change(3)
    out["ret_6"] = c.pct_change(6)
    out["volatility_20"] = out["log_return"].rolling(20, min_periods=10).std()

    atr = _atr(h, l, c)
    out["natr"] = (atr / (c + eps)) * 100.0
    out["range_pct"] = (h - l) / (c + eps) * 100.0

    # ================= ACAO DO PRECO =================
    body = (c - o)
    out["body_pct"] = body / (c + eps) * 100.0
    out["close_pos"] = (c - l) / (h - l + eps)          # 0=fundo da vela, 1=topo
    out["upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (atr + eps)
    out["lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (atr + eps)

    # ================= TENDENCIA =================
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=20).mean()
    out["ema_fast_ratio"] = ema8 / (ema21 + eps) - 1.0
    out["ema_trend_ratio"] = ema21 / (ema50 + eps) - 1.0
    out["price_vs_ema21"] = c / (ema21 + eps) - 1.0
    out["ema21_slope"] = ema21.pct_change(3)
    out["adx"] = _adx(h, l, c)

    # ================= MOMENTO =================
    out["rsi"] = _rsi(c)
    out["rsi_slope"] = out["rsi"].diff(3)
    out["stoch_k"] = _stoch_k(h, l, c)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    out["macd_hist"] = _macd_hist(c)
    out["macd_hist_slope"] = out["macd_hist"].diff(2)
    out["roc_6"] = c.pct_change(6) * 100.0

    # ================= VOLATILIDADE ESTRUTURAL =================
    sma20 = c.rolling(20, min_periods=10).mean()
    std20 = c.rolling(20, min_periods=10).std(ddof=0)
    bb_up = sma20 + 2 * std20
    bb_lo = sma20 - 2 * std20
    out["bb_pos"] = (c - bb_lo) / (bb_up - bb_lo + eps)
    out["bb_width"] = (bb_up - bb_lo) / (sma20 + eps)

    kc_up = ema21 + 1.5 * atr
    kc_lo = ema21 - 1.5 * atr
    out["squeeze"] = (bb_up - bb_lo) / (kc_up - kc_lo + eps)

    # ================= VOLUME / FLUXO =================
    vol_mean20 = v.rolling(20, min_periods=5).mean()
    vol_std20 = v.rolling(20, min_periods=5).std(ddof=0)
    out["vol_z"] = (v - vol_mean20) / (vol_std20 + eps)
    out["vol_ratio"] = v / (v.rolling(5).mean() + eps)

    candle_dir = np.sign(c - o).replace(0, np.nan).ffill().fillna(0.0)
    cvd = (v * candle_dir).cumsum()
    cvd_base = cvd.rolling(20, min_periods=5).mean()
    out["cvd_slope"] = (cvd - cvd_base) / (cvd_base.abs() + eps)

    obv = (np.sign(c.diff()) * v).fillna(0.0).cumsum()
    out["obv_slope"] = obv.diff(5) / (obv.rolling(50).std() + eps)

    # ================= DERIVATIVOS — OI + FUNDING =================
    out["oi_change"] = oi.pct_change().replace([np.inf, -np.inf], np.nan)
    oi_mean20 = oi.rolling(20, min_periods=5).mean()
    oi_std20 = oi.rolling(20, min_periods=5).std(ddof=0)
    out["oi_z"] = (oi - oi_mean20) / (oi_std20 + eps)
    oi_ema8 = oi.ewm(span=8, adjust=False).mean()
    oi_ema21 = oi.ewm(span=21, adjust=False).mean()
    out["oi_trend"] = oi_ema8 / (oi_ema21 + eps) - 1.0

    ret_5 = c.pct_change(5)
    oi_chg_5 = oi.pct_change(5)
    out["oi_price_div"] = np.where(
        (ret_5 > 0) & (oi_chg_5 > 0), 1.0,        # alta + OI subindo = longs abrindo
        np.where((ret_5 > 0) & (oi_chg_5 < 0), -1.0,   # alta + OI caindo = shorts fechando (fraco)
        np.where((ret_5 < 0) & (oi_chg_5 > 0), -0.5,   # queda + OI subindo = shorts abrindo
        np.where((ret_5 < 0) & (oi_chg_5 < 0), 0.5, 0.0))))  # queda + OI caindo = longs fechando

    out["funding"] = fr
    out["funding_ema"] = fr.ewm(span=8, adjust=False).mean()
    out["funding_delta"] = fr.diff(3)

    # ================= MULTI-TIMEFRAME (so passado, via resample+shift+ffill) =================
    dt = pd.to_datetime(out["timestamp"], unit="ms", utc=True)

    mtf1h = _resample_mtf(out, "1h")
    mtf4h = _resample_mtf(out, "4h")

    def _mtf_cols(mtf: pd.DataFrame, prefix: str) -> pd.DataFrame:
        aligned = mtf.reindex(dt).ffill()
        dist = (c.values / (aligned["ema20"].values + eps)) - 1.0
        return pd.DataFrame({
            f"{prefix}_dist": dist,
            f"{prefix}_rsi": aligned["rsi"].values,
            f"{prefix}_macdh": aligned["macdh"].values,
        }, index=out.index)

    mtf_feat = pd.concat([_mtf_cols(mtf1h, "mtf1h"), _mtf_cols(mtf4h, "mtf4h")], axis=1)
    out = pd.concat([out, mtf_feat], axis=1)

    # ================= CONTEXTO BTC =================
    if btc_df is not None and not btc_df.empty:
        btc = btc_df[["timestamp", "close"]].copy()
        btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
        btc["btc_close"] = btc["close"]
        out = out.merge(btc[["timestamp", "btc_close"]], on="timestamp", how="left")
        out["btc_close"] = out["btc_close"].ffill()
        btc_ret = np.log(out["btc_close"] / out["btc_close"].shift(1).replace(0, np.nan))
        out["btc_corr"] = out["log_return"].rolling(20, min_periods=10).corr(btc_ret)
        # Forca relativa de 24 barras (~2h em 5m): ativo vs BTC
        out["rel_strength_24"] = (
            (c / c.shift(24).replace(0, np.nan))
            - (out["btc_close"] / out["btc_close"].shift(24).replace(0, np.nan))
        )
        out.drop(columns=["btc_close"], inplace=True)
    else:
        out["btc_corr"] = 1.0
        out["rel_strength_24"] = 0.0

    # ================= RELOGIO DO MERCADO =================
    hours = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ================= HIGIENIZACAO =================
    out = out.replace([np.inf, -np.inf], np.nan)

    # Descarta warmup: convergencia das janelas e do resample MTF
    if len(out) <= WARMUP_BARS:
        raise ValueError(
            f"features.py: insuficiente ({len(out)} barras). "
            f"Minimo: WARMUP_BARS + 1 = {WARMUP_BARS + 1}."
        )
    out = out.iloc[WARMUP_BARS:].copy()

    # Colunas de entrada numericas ficam intactas; features sem historico -> 0.0
    faltantes = [f for f in FEATURE_NAMES if f not in out.columns]
    if faltantes:
        raise ValueError(f"features.py: features nao geradas: {faltantes}")

    out[FEATURE_NAMES] = out[FEATURE_NAMES].ffill().fillna(0.0)
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------
# REGIME HMM — entradas produzidas aqui, modelo treinado no treinador
# ---------------------------------------------------------------------
def aplicar_regime_hmm(df_feat: pd.DataFrame, hmm_model) -> pd.Series:
    """
    Aplica HMM ja treinado sobre as colunas HMM_INPUTS.
    No treino: fit no treino, predict no treino/validacao.
    Na inferencia: predict da ultima linha com o HMM salvo no .pkl.
    Retorna Series int (estado) alinhada ao indice de df_feat.
    """
    X = df_feat[HMM_INPUTS].fillna(0.0).values
    try:
        states = hmm_model.predict(X)
    except Exception:
        states = np.zeros(len(df_feat), dtype=int)
    return pd.Series(states, index=df_feat.index, name="hmm_regime")
