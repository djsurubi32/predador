"""
rotulador.py — Barreira de primeiro toque do Predador v3.2.

Para cada vela i do historico, responde:
    "O preco tocou +BARRIER_PCT% antes de -BARRIER_PCT%,
     dentro das proximas LABEL_HORIZON_CANDLES velas?"

Rotulo:
    1 = tocou a barreira positiva primeiro (momento de alta)
    0 = tocou a barreira negativa primeiro (momento de baixa)
    descartada = nenhuma barreira tocada no horizonte (mercado parado)

Simetrico e percentual: identico para BTC, memecoin e altcoin.

Contrato de entrada (saida de features.py, que ja descartou o warmup):
    close, high, low (numpy/pandas, sem NaN nas colunas de preco)

Saida:
    Series int64 (1/0) alinhada ao indice de entrada, so com as
    velas rotuladas (as descartadas sao removidas).

Uso:
    y = rotular(df_feat["close"], df_feat["high"], df_feat["low"])
    df_treino = df_feat.loc[y.index]
"""

import logging
import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def rotular(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    barrier_pct: float = None,
    horizon: int = None,
) -> pd.Series:
    """
    Rotula cada vela com a barreira de primeiro toque simetrica.

    Args:
        close, high, low: series de precos alinhadas (sem NaN no fim).
        barrier_pct: distancia das barreiras em % (default: Config.BARRIER_PCT).
        horizon: quantas velas a frente olhar (default: Config.LABEL_HORIZON_CANDLES).

    Returns:
        Series int64 com rotulos 1/0, index = das velas rotuladas.
        Velas cujo futuro ficou entre as barreiras NAO aparecem no resultado.
    """
    if barrier_pct is None:
        barrier_pct = Config.BARRIER_PCT
    if horizon is None:
        horizon = Config.LABEL_HORIZON_CANDLES

    if barrier_pct <= 0:
        raise ValueError(f"rotulador: BARRIER_PCT invalido ({barrier_pct}).")
    if horizon < 1:
        raise ValueError(f"rotulador: horizonte invalido ({horizon}).")

    c = close.to_numpy(dtype=np.float64)
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    n = len(c)

    if n < horizon + 10:
        raise ValueError(f"rotulador: historico curto demais ({n} velas).")

    idx = close.index.to_numpy()
    rotulos = np.full(n, -1, dtype=np.int8)  # -1 = descartada por padrao

    fator = barrier_pct / 100.0

    # Vetorizado: para cada vela i, varre o futuro ate o horizonte.
    for i in range(n - horizon):
        entry = c[i]
        if entry <= 0 or np.isnan(entry):
            continue

        barreira_pos = entry * (1.0 + fator)
        barreira_neg = entry * (1.0 - fator)

        toque_pos = -1   # indice relativo do primeiro toque na barreira +
        toque_neg = -1   # indice relativo do primeiro toque na barreira -

        for j in range(1, horizon + 1):
            k = i + j
            if h[k] >= barreira_pos and toque_pos < 0:
                toque_pos = j
            if l[k] <= barreira_neg and toque_neg < 0:
                toque_neg = j
            if toque_pos > 0 and toque_neg > 0:
                break  # ambas tocadas, decide abaixo

        if toque_pos < 0 and toque_neg < 0:
            continue  # nenhuma tocada no horizonte -> descarta

        if toque_pos > 0 and (toque_neg < 0 or toque_pos < toque_neg):
            rotulos[i] = 1
        elif toque_neg > 0 and (toque_pos < 0 or toque_neg < toque_pos):
            rotulos[i] = 0
        else:
            # Empate: mesma vela futura (k) tocou as duas (alta volatilidade).
            # Avalia o fechamento daquela vela futura k em relacao a entrada:
            # conservador — precisa fechar com folga positiva para marcar 1.
            k_empate = i + toque_pos
            rotulos[i] = 1 if c[k_empate] > entry * (1.0 + fator * 0.5) else 0

    mascara = rotulos >= 0
    y = pd.Series(rotulos[mascara].astype(np.int64),
                  index=pd.Index(idx[mascara]), name="target")

    if len(y) == 0:
        raise ValueError(
            "rotulador: ZERO velas rotuladas. BARRIER_PCT pode estar alto "
            "demais para a volatilidade do ativo/timeframe."
        )

    return y


def estatisticas_rotulo(y: pd.Series, symbol: str = "") -> dict:
    """
    Metricas de qualidade do rotulo — loga e retorna dict.
    Usado pelo treinador para validar se o ativo gera sinal suficiente.
    """
    total = len(y)
    positivos = int((y == 1).sum())
    taxa_pos = positivos / total if total else 0.0

    stats = {
        "symbol": symbol,
        "rotuladas": total,
        "taxa_positivos": round(taxa_pos, 4),
    }

    logger.info(
        f"[ROTULADOR] {symbol}: {total} velas rotuladas | "
        f"{taxa_pos:.1%} positivas (toque +{Config.BARRIER_PCT}% primeiro)"
    )

    # Alerta operacional: taxa muito desequilibrada indica que a barreira
    # e assimetrica em relacao a volatilidade do ativo.
    if taxa_pos < 0.30 or taxa_pos > 0.70:
        logger.warning(
            f"[ROTULADOR] {symbol}: taxa de positivos fora de 30-70% "
            f"({taxa_pos:.1%}). Verifique se BARRIER_PCT={Config.BARRIER_PCT}% "
            f"esta adequado a volatilidade deste ativo."
        )

    return stats
