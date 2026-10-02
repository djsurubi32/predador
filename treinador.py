"""
treinador.py — Motor de treinamento do Predador v3.2.

FLUXO POR SIMBOLO:
    1. Baixa OHLCV 5m + Open Interest + Funding (Bybit, ~100k velas)
    2. Monta benchmark BTC real
    3. features.adicionar_features() — MESMO pipeline da inferencia
    4. HMM fit somente no treino (sem leakage de regime)
    5. rotulador.rotular() — barreira +/-BARRIER_PCT nas proximas 2 velas
    6. Split temporal com embargo (purged)
    7. Stacking: LGBM + XGB + CatBoost -> RandomForest meta
    8. Calibracao de probabilidade (o gate de 65% so funciona calibrado)
    9. Persiste .pkl com o contrato do analisador

CONTRATO DO .pkl (consumido pelo analisador.py):
    payload = {
        "modelo":          estimator com predict_proba (calibrado),
        "feature_names":   lista de colunas na ordem exata do treino,
        "hmm":             modelo HMM treinado (para regime na inferencia),
        "symbol":          "BTC/USDT:USDT",
        "barrier_pct":     0.20,
        "label_horizon":   2,
        "timeframe":       "5m",
        "metrics":         dict com logloss/brier/gate/cobertura/expectativa,
        "trained_at":      timestamp,
    }
"""

import os
import gc
import time
import logging
import warnings
import asyncio
import joblib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

from config import Config, universe_provider
from features import (
    adicionar_features,
    aplicar_regime_hmm,
    FEATURE_NAMES,
    HMM_INPUTS,
)
from rotulador import rotular, estatisticas_rotulo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [TREINADOR] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

TF_SECONDS = {"5m": 300, "1m": 60, "15m": 900, "1h": 3600}

MIN_VELAS_ROTULADAS = 500       # abaixo disso, simbolo nao gera modelo
GATE_AVALIACAO = Config.MIN_PROB_ENTRY  # 65.0


class DummyHMM:
    """Fallback se o HMM nao convergir."""
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class CalibratedStackingEnsemble:
    """
    Pipeline completo do Stacking:
    Recebe as features brutas (49 colunas), extrai as probabilidades dos modelos
    base (LGBM + XGB + CatBoost) e as submete ao calibrador isotonico.
    """
    def __init__(self, lgbm, xgbm, catb, calibrator):
        self.lgbm = lgbm
        self.xgbm = xgbm
        self.catb = catb
        self.calibrator = calibrator

    def predict_proba(self, X):
        X_stack = np.column_stack([
            self.lgbm.predict_proba(X),
            self.xgbm.predict_proba(X),
            self.catb.predict_proba(X),
        ])
        return self.calibrator.predict_proba(X_stack)


# =====================================================================
# FETCH DE DADOS (OHLCV + OI + Funding) — Bybit publica/testnet
# =====================================================================
async def fetch_historico_completo(exchange, symbol: str, limit: int) -> list:
    """
    Retorna lista de barras [ts, o, h, l, c, v, open_interest, funding].
    OI e funding sao alinhados por timestamp (forward-fill entre amostras).
    """
    try:
        tf_sec = TF_SECONDS.get(Config.TIMEFRAME, 300)
        todas = []
        since_ms = int((time.time() - (limit * tf_sec)) * 1000)

        while len(todas) < limit:
            lote = await exchange.fetch_ohlcv(
                symbol, Config.TIMEFRAME, since=since_ms, limit=1000
            )
            if not lote:
                break
            since_ms = lote[-1][0] + 1
            todas.extend(lote)
            if len(lote) < 1000:
                break
        todas = todas[-limit:]
        if len(todas) < 1000:
            return []

        # ---- Open Interest ----
        oi_map = {}
        try:
            oi_data = await exchange.fetch_open_interest_history(
                symbol, Config.TIMEFRAME, limit=min(1000, limit)
            )
            for item in oi_data:
                ts = int(item.get("timestamp", 0))
                val = float(
                    item.get("openInterestValue")
                    or (item.get("info", {}) or {}).get("openInterest", 0)
                    or 0
                )
                if val > 0:
                    oi_map[ts] = val
        except Exception as e:
            logging.debug(f"OI indisponivel {symbol}: {e}")

        # ---- Funding Rate ----
        fr_map = {}
        try:
            fr_data = await exchange.fetch_funding_rate_history(
                symbol, limit=min(1000, limit)
            )
            for item in fr_data:
                ts = int(item.get("timestamp", 0))
                fr_map[ts] = float(item.get("fundingRate", 0) or 0)
        except Exception as e:
            logging.debug(f"Funding indisponivel {symbol}: {e}")

        # ---- Merge forward-fill ----
        merged = []
        last_oi, last_fr = 0.0, 0.0
        for bar in todas:
            ts = int(bar[0])
            if oi_map.get(ts, 0.0) != 0.0:
                last_oi = oi_map[ts]
            if ts in fr_map:
                last_fr = fr_map[ts]
            merged.append(
                [bar[0], bar[1], bar[2], bar[3], bar[4], bar[5], last_oi, last_fr]
            )
        return merged

    except Exception as e:
        logging.error(f"Erro ao baixar {symbol}: {e}")
        return []


# =====================================================================
# NUCLEO DE TREINO (CPU-bound — roda em thread pool)
# =====================================================================
def treinar_simbolo(symbol: str, bars: list, btc_bars: list) -> str:
    if len(bars) < 2000:
        return f"PULADO {symbol}: apenas {len(bars)} barras."

    df = pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close",
                 "volume", "open_interest", "funding_rate"],
    )
    btc_df = (
        pd.DataFrame(btc_bars, columns=["timestamp", "open", "high", "low",
                                        "close", "volume", "open_interest", "funding_rate"])
        if btc_bars else None
    )

    # ---- Features (pipeline unico) ----
    try:
        df_feat = adicionar_features(df, btc_df[["timestamp", "close"]] if btc_df is not None else None)
    except ValueError as e:
        return f"PULADO {symbol}: {e}"

    # ---- Rotulos ----
    try:
        y = rotular(df_feat["close"], df_feat["high"], df_feat["low"])
    except ValueError as e:
        return f"PULADO {symbol}: {e}"

    stats = estatisticas_rotulo(y, symbol)
    if stats["rotuladas"] < MIN_VELAS_ROTULADAS:
        return (
            f"PULADO {symbol}: so {stats['rotuladas']} velas rotuladas "
            f"(minimo {MIN_VELAS_ROTULADAS})."
        )

    df_t = df_feat.loc[y.index].copy()
    yy = y.loc[df_t.index]

    if len(np.unique(yy)) < 2:
        return f"PULADO {symbol}: rotulo sem contraste (so uma classe)."

    # ---- Split temporal com embargo (purged pelo horizonte do rotulo) ----
    n = len(df_t)
    embargo = max(Config.LABEL_HORIZON_CANDLES + 1,
                  int(n * Config.PURGED_CV_EMBARGO_PCT))
    corte_meta = int(n * 0.75)   # 75% treino | 12.5% meta | 12.5% calibracao
    corte_cal = int(n * 0.875)

    if corte_meta - embargo < 500 or (n - corte_cal) < 200:
        return f"PULADO {symbol}: historico insuficiente para os 3 blocos."

    df_train = df_t.iloc[: corte_meta - embargo]
    y_train = yy.iloc[: corte_meta - embargo]
    df_meta = df_t.iloc[corte_meta:corte_cal]
    y_meta = yy.iloc[corte_meta:corte_cal]
    df_cal = df_t.iloc[corte_cal:]
    y_cal = yy.iloc[corte_cal:]

    # ---- HMM: fit SOMENTE no treino (sem leakage) ----
    hmm = GaussianHMM(
        n_components=3, covariance_type="diag", n_iter=100,
        random_state=42, min_covar=1e-3,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(df_train[HMM_INPUTS].fillna(0.0).values)
        for d, nome in ((df_train, "train"), (df_meta, "meta"), (df_cal, "cal")):
            d["hmm_regime"] = aplicar_regime_hmm(d, hmm).values
    except Exception:
        hmm = DummyHMM()
        df_train["hmm_regime"] = 0
        df_meta["hmm_regime"] = 0
        df_cal["hmm_regime"] = 0

    feature_cols = FEATURE_NAMES + ["hmm_regime"]
    faltantes = [f for f in feature_cols if f not in df_train.columns]
    if faltantes:
        return f"ERRO {symbol}: features ausentes {faltantes}"

    X_train, X_meta = df_train[feature_cols], df_meta[feature_cols]
    X_cal = df_cal[feature_cols]

    # ---- Modelos base ----
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
        class_weight="balanced", reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=2,
    )
    xgbm = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        early_stopping_rounds=20, random_state=42,
        eval_metric="logloss", n_jobs=2,
    )
    catb = CatBoostClassifier(
        iterations=300, learning_rate=0.03, depth=5,
        auto_class_weights="Balanced", l2_leaf_reg=3.0,
        early_stopping_rounds=20, silent=True, random_state=42, thread_count=2,
    )

    lgbm.fit(X_train, y_train)
    xgbm.fit(X_train, y_train, eval_set=[(X_meta, y_meta)], verbose=False)
    catb.fit(X_train, y_train, eval_set=(X_meta, y_meta), verbose=False)

    # ---- Meta-learner: RF sobre probabilidades dos base ----
    X_meta_stack = np.column_stack([
        lgbm.predict_proba(X_meta),
        xgbm.predict_proba(X_meta),
        catb.predict_proba(X_meta),
    ])
    meta = RandomForestClassifier(
        n_estimators=100, max_depth=4, class_weight="balanced",
        random_state=42, n_jobs=2, min_samples_leaf=20,
    )
    meta.fit(X_meta_stack, y_meta)

    # ---- Calibracao (cv='prefit' sobre bloco exclusivo) ----
    X_cal_stack = np.column_stack([
        lgbm.predict_proba(X_cal),
        xgbm.predict_proba(X_cal),
        catb.predict_proba(X_cal),
    ])
    calibrator = CalibratedClassifierCV(meta, cv="prefit", method="isotonic")
    calibrator.fit(X_cal_stack, y_cal)

    # Cria o modelo final integrado com as 49 features
    pipeline_modelo = CalibratedStackingEnsemble(lgbm, xgbm, catb, calibrator)

    # ---- Metricas no bloco de calibracao ----
    prob_cal = pipeline_modelo.predict_proba(X_cal)[:, 1]
    ll = log_loss(y_cal, np.column_stack([1 - prob_cal, prob_cal]))
    brier = brier_score_loss(y_cal, prob_cal)

    gate_mask = prob_cal >= (GATE_AVALIACAO / 100.0)
    cobertura = float(gate_mask.mean())
    if gate_mask.sum() >= 30:
        precisao_gate = float((y_cal.values[gate_mask] == 1).mean())
    else:
        precisao_gate = float("nan")

    # Expectativa em "R" (risco=1, ganho=1), descontando custos maker round-trip
    custo_pct = (Config.MAKER_FEE_PCT + Config.SLIPPAGE_MAKER_PCT) * 2
    custo_em_r = custo_pct / Config.BARRIER_PCT  # custo como fracao da barreira
    if gate_mask.sum() >= 30:
        p = precisao_gate
        expectativa_r = p * (1 - custo_em_r) - (1 - p) * (1 + custo_em_r)
    else:
        expectativa_r = float("nan")

    metrics = {
        "logloss": round(float(ll), 4),
        "brier": round(float(brier), 4),
        "precisao_no_gate_65": round(precisao_gate, 4) if precisao_gate == precisao_gate else None,
        "cobertura_gate_65": round(cobertura, 4),
        "expectativa_r_no_gate": round(expectativa_r, 4) if expectativa_r == expectativa_r else None,
        "n_train": int(len(X_train)),
        "n_meta": int(len(X_meta)),
        "n_cal": int(len(X_cal)),
        "embargo_barras": int(embargo),
    }

    # ---- PERSISTENCIA — contrato com o analisador ----
    payload = {
        "modelo": pipeline_modelo,     # predict_proba(X_49) -> P(tocar +barreira)
        "feature_names": feature_cols, # ordem exata
        "hmm": hmm,
        "symbol": symbol,
        "barrier_pct": Config.BARRIER_PCT,
        "label_horizon": Config.LABEL_HORIZON_CANDLES,
        "timeframe": Config.TIMEFRAME,
        "metrics": metrics,
        "trained_at": int(time.time()),
    }

    nome_seguro = symbol.replace("/", "_").replace(":", "_")
    caminho = Path(Config.MODELS_DIR) / f"{nome_seguro}.pkl"
    joblib.dump(payload, caminho)

    return (
        f"OK {symbol} | logloss={ll:.4f} brier={brier:.4f} | "
        f"gate65: prec={precisao_gate:.1%} cov={cobertura:.1%} "
        f"E={expectativa_r:+.3f}R | train={len(X_train)} cal={len(X_cal)}"
        if precisao_gate == precisao_gate else
        f"OK {symbol} | logloss={ll:.4f} | gate65: amostra insuficiente "
        f"({int(gate_mask.sum())} trades) — cobertura {cobertura:.1%}"
    )


# =====================================================================
# MOTOR ASSINCRONO
# =====================================================================
class MotorTreinamento:
    def __init__(self):
        self.exchange = ccxt_async.bybit(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        self.btc_symbol = "BTC/USDT:USDT"

    async def iniciar_ciclo(self):
        ativos = await universe_provider.get_ativos()
        logging.info(f"Forja iniciada: {len(ativos)} ativos | tf={Config.TIMEFRAME} "
                     f"| barreira +/-{Config.BARRIER_PCT}% | horizonte {Config.LABEL_HORIZON_CANDLES} velas")

        btc_bars = await fetch_historico_completo(
            self.exchange, self.btc_symbol, Config.CANDLES_TREINAMENTO
        )
        if not btc_bars:
            logging.error("Benchmark BTC indisponivel — abortando ciclo.")
            return

        loop = asyncio.get_running_loop()
        ok = falhas = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            for symbol in ativos:
                logging.info(f"Baixando {symbol}...")
                bars = await fetch_historico_completo(
                    self.exchange, symbol, Config.CANDLES_TREINAMENTO
                )
                if not bars:
                    logging.warning(f"Sem dados para {symbol}.")
                    falhas += 1
                    continue
                try:
                    msg = await loop.run_in_executor(
                        pool, treinar_simbolo, symbol, bars, btc_bars
                    )
                    logging.info(f"Treino {symbol}: {msg}")
                    ok += 1 if msg.startswith("OK") else 0
                    falhas += 0 if msg.startswith("OK") else 1
                except Exception as e:
                    logging.error(f"Falha fatal {symbol}: {e}")
                    falhas += 1
                del bars
                gc.collect()

        logging.info(
            f"Ciclo concluido: {ok} modelos | {falhas} pulados/falhas. "
            f"Proximo ciclo em {Config.HORAS_RETREINO}h."
        )


async def main():
    motor = MotorTreinamento()
    try:
        while True:
            try:
                await motor.iniciar_ciclo()
            except Exception as e:
                logging.error(f"Erro no ciclo: {e}")
            await asyncio.sleep(Config.HORAS_RETREINO * 3600)
    finally:
        await motor.exchange.close()


if __name__ == "__main__":
    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Treinador encerrado.")
