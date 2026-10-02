"""
backtester.py — Validacao de edge do Predador v3.2.

METODOLOGIA (sem auto-engano):
    1. Baixa historico real (OHLCV + OI + funding) da Bybit
    2. features.adicionar_features() — MESMO pipeline da producao
    3. rotulador.rotular() — MESMO rotulo da producao
    4. Split temporal PURGADO com embargo (nada de aleatorio)
       42% treino | 14% meta (stacking) | 14% calibracao
       30% final = teste out-of-sample (o modelo nunca viu)
    5. Pipeline de treino MIRROR do treinador.py (LGBM+XGB+Cat -> RF meta
       -> calibracao isotonica). Se mudar o treinador, mudar aqui.
    6. Simulacao evento a evento no teste:
       - Sinal na FECHAMENTO da vela i (igual ao analisador)
       - Entrada na ABERTURA da vela i+1 (realista)
       - SL/TP nas barreiras exatas +/-BARRIER_PCT%
       - Se na mesma vela bate SL e TP: conta STOP (conservador)
       - Time-stop: fecha no fechamento da 3a vela (15 min)
       - Custos: maker na entrada; maker se sair por SL/TP, taker se time-stop
       - 1 posicao por vez (igual ao gerenciador)
       - Gate de 65% calibrado

GATE DE APROVACAO (para ir a testnet):
    - minimo 200 trades out-of-sample
    - Profit Factor > 1.30
    - Expectativa > +0.10R por trade
    - Max drawdown < 15R

USO:
    python backtester.py BTC/USDT:USDT ETH/USDT:USDT
    python backtester.py --universo     (top 5 do universo Config)
"""

import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from config import Config, universe_provider
from features import adicionar_features, aplicar_regime_hmm, FEATURE_NAMES, HMM_INPUTS
from rotulador import rotular, estatisticas_rotulo
from treinador import fetch_historico_completo, CalibratedStackingEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [BACKTESTER] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

REPORT_DIR = Path(__file__).resolve().parent / "backtests"
REPORT_DIR.mkdir(exist_ok=True)

MIN_TRADES_OOS = 200
GATE_PF = 1.30
GATE_EXPECTATIVA = 0.10
GATE_MAX_DD_R = 15.0


class DummyHMM:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# =====================================================================
# TREINO — espelho fiel do treinador.py
# =====================================================================
def treinar_pipeline(df_train, y_train, df_meta, y_meta, df_cal, y_cal):
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100,
                      random_state=42, min_covar=1e-3)
    try:
        hmm.fit(df_train[HMM_INPUTS].fillna(0.0).values)
        for d in (df_train, df_meta, df_cal):
            d["hmm_regime"] = aplicar_regime_hmm(d, hmm).values
    except Exception:
        hmm = DummyHMM()
        for d in (df_train, df_meta, df_cal):
            d["hmm_regime"] = 0

    cols = FEATURE_NAMES + ["hmm_regime"]
    X_tr, X_me, X_ca = df_train[cols], df_meta[cols], df_cal[cols]

    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6,
                              num_leaves=31, class_weight="balanced", reg_alpha=0.1,
                              reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=2)
    xgbm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5,
                             early_stopping_rounds=20, random_state=42,
                             eval_metric="logloss", n_jobs=2)
    catb = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5,
                              auto_class_weights="Balanced", l2_leaf_reg=3.0,
                              early_stopping_rounds=20, silent=True,
                              random_state=42, thread_count=2)

    lgbm.fit(X_tr, y_train)
    xgbm.fit(X_tr, y_train, eval_set=[(X_me, y_meta)], verbose=False)
    catb.fit(X_tr, y_train, eval_set=(X_me, y_meta), verbose=False)

    X_me_stack = np.column_stack([lgbm.predict_proba(X_me),
                                  xgbm.predict_proba(X_me),
                                  catb.predict_proba(X_me)])
    meta = RandomForestClassifier(n_estimators=100, max_depth=4,
                                  class_weight="balanced", random_state=42,
                                  n_jobs=2, min_samples_leaf=20)
    meta.fit(X_me_stack, y_meta)

    X_ca_stack = np.column_stack([lgbm.predict_proba(X_ca),
                                  xgbm.predict_proba(X_ca),
                                  catb.predict_proba(X_ca)])
    cal = CalibratedClassifierCV(meta, cv="prefit", method="isotonic")
    cal.fit(X_ca_stack, y_cal)

    # Pipeline integrado: X (49 features) -> base models -> meta/calibrator
    pipeline = CalibratedStackingEnsemble(lgbm, xgbm, catb, cal)
    modelo_final = lambda X: pipeline.predict_proba(X)[:, 1]

    return modelo_final, hmm, cols


# =====================================================================
# SIMULACAO EVENTO A EVENTO
# =====================================================================
def simular(df_test: pd.DataFrame, prob_up: np.ndarray, symbol: str) -> dict:
    o = df_test["open"].to_numpy(float)
    h = df_test["high"].to_numpy(float)
    l = df_test["low"].to_numpy(float)
    c = df_test["close"].to_numpy(float)
    n = len(df_test)

    gate = Config.MIN_PROB_ENTRY / 100.0
    fator = Config.BARRIER_PCT / 100.0
    ts_barras = max(1, Config.TIME_STOP_MINUTES // 5)  # 3 velas

    custo_maker = (Config.MAKER_FEE_PCT + Config.SLIPPAGE_MAKER_PCT) / 100.0
    custo_taker = (Config.TAKER_FEE_PCT + Config.SLIPPAGE_TAKER_PCT) / 100.0
    custo_maker_r = custo_maker / fator   # custo em multiplos da barreira
    custo_taker_r = custo_taker / fator

    trades = []
    i = 0
    while i < n - 1:
        p = prob_up[i]
        direcao = None
        if p >= gate:
            direcao = "BUY"
        elif (1.0 - p) >= gate:
            direcao = "SELL"

        if direcao is None:
            i += 1
            continue

        # Entrada na abertura da proxima vela (realista)
        entry = o[i + 1]
        if entry <= 0:
            i += 1
            continue
        sl_p = entry * (1 - fator) if direcao == "BUY" else entry * (1 + fator)
        tp_p = entry * (1 + fator) if direcao == "BUY" else entry * (1 - fator)

        resultado_r = None
        saida_preco = None
        motivo = None
        j_fim = min(i + 1 + ts_barras, n - 1)

        for j in range(i + 1, j_fim + 1):
            hit_sl = l[j] <= sl_p if direcao == "BUY" else h[j] >= sl_p
            hit_tp = h[j] >= tp_p if direcao == "BUY" else l[j] <= tp_p

            if hit_sl and hit_tp:
                resultado_r = -1.0 - custo_maker_r
                saida_preco = sl_p
                motivo = "SL_ambiguo"
                i = j
                break
            if hit_sl:
                resultado_r = -1.0 - custo_maker_r
                saida_preco = sl_p
                motivo = "SL"
                i = j
                break
            if hit_tp:
                resultado_r = 1.0 - custo_maker_r
                saida_preco = tp_p
                motivo = "TP"
                i = j
                break

            if j == j_fim:
                # Time-stop no fechamento
                if direcao == "BUY":
                    raw = (c[j] - entry) / (entry * fator)
                else:
                    raw = (entry - c[j]) / (entry * fator)
                resultado_r = raw - custo_taker_r
                saida_preco = c[j]
                motivo = "TIME_STOP"
                i = j
                break
        else:
            i = j_fim
            continue

        trades.append({
            "i_entrada": i + 1,
            "ts": int(df_test["timestamp"].iloc[min(i, n - 1)]),
            "direcao": direcao,
            "p_sinal": float(max(p, 1 - p)),
            "resultado_r": round(resultado_r, 4),
            "motivo": motivo,
        })

    if not trades:
        return {"symbol": symbol, "trades": 0, "aprovado": False,
                "motivo": "zero_trades_no_teste"}

    r = np.array([t["resultado_r"] for t in trades])
    wins = r[r > 0]
    losses = r[r < 0]

    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    winrate = float((r > 0).mean())
    expectativa = float(r.mean())

    # Drawdown em R
    eq = np.cumsum(r)
    pico = np.maximum.accumulate(eq)
    dd = float((pico - eq).max())

    motivos = pd.Series([t["motivo"] for t in trades]).value_counts().to_dict()

    aprovado = (len(trades) >= MIN_TRADES_OOS and pf >= GATE_PF
                and expectativa >= GATE_EXPECTATIVA and dd <= GATE_MAX_DD_R)

    return {
        "symbol": symbol,
        "trades": len(trades),
        "winrate": round(winrate, 4),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "expectativa_r": round(expectativa, 4),
        "max_dd_r": round(dd, 2),
        "total_r": round(float(eq[-1]), 2),
        "distribuicao_saidas": motivos,
        "aprovado": aprovado,
        "motivo": ("OK" if aprovado else
                   f"gate: trades>={MIN_TRADES_OOS} & PF>={GATE_PF} "
                   f"& E>={GATE_EXPECTATIVA}R & DD<={GATE_MAX_DD_R}R"),
        "trades_detalhe": trades,
    }


# =====================================================================
# PIPELINE POR SIMBOLO
# =====================================================================
async def backtest_symbol(exchange, symbol: str, btc_bars: list) -> dict:
    logging.info(f"--- {symbol} ---")
    bars = await fetch_historico_completo(exchange, symbol, Config.CANDLES_TREINAMENTO)
    if len(bars) < 5000:
        return {"symbol": symbol, "aprovado": False, "motivo": "historico insuficiente"}

    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close",
                                     "volume", "open_interest", "funding_rate"])
    btc_df = (pd.DataFrame(btc_bars, columns=["timestamp", "open", "high", "low",
                                              "close", "volume", "open_interest",
                                              "funding_rate"])[["timestamp", "close"]]
              if btc_bars else None)

    try:
        df_feat = adicionar_features(df, btc_df)
    except ValueError as e:
        return {"symbol": symbol, "aprovado": False, "motivo": str(e)}

    try:
        y = rotular(df_feat["close"], df_feat["high"], df_feat["low"])
    except ValueError as e:
        return {"symbol": symbol, "aprovado": False, "motivo": str(e)}

    estatisticas_rotulo(y, symbol)
    df_t = df_feat.loc[y.index].copy()
    yy = y.loc[df_t.index].to_numpy()

    n = len(df_t)
    embargo = max(Config.LABEL_HORIZON_CANDLES + 1, int(n * Config.PURGED_CV_EMBARGO_PCT))

    # 42% treino | 14% meta | 14% cal | 30% teste (do total rotulado)
    c1 = int(n * 0.42)
    c2 = int(n * 0.56)
    c3 = int(n * 0.70)

    if c1 - embargo < 500 or (n - c3) < 500:
        return {"symbol": symbol, "aprovado": False,
                "motivo": "amostra insuficiente para split 42/14/14/30"}

    idx = df_t.index
    df_train, y_train = df_t.loc[idx[: c1 - embargo]], yy[: c1 - embargo]
    df_meta, y_meta = df_t.loc[idx[c1:c2]], yy[c1:c2]
    df_cal, y_cal = df_t.loc[idx[c2:c3]], yy[c2:c3]
    df_test = df_t.loc[idx[c3:]].copy()

    logging.info(f"Split: train={len(df_train)} meta={len(df_meta)} "
                 f"cal={len(df_cal)} TESTE={len(df_test)}")

    modelo_prob, hmm, cols = treinar_pipeline(df_train, y_train, df_meta, y_meta,
                                              df_cal, y_cal)
    df_test["hmm_regime"] = aplicar_regime_hmm(df_test, hmm).values
    prob_up = modelo_prob(df_test[cols].fillna(0.0))

    brier = brier_score_loss(yy[c3:], prob_up)

    resultado = simular(df_test.reset_index(drop=True), np.asarray(prob_up), symbol)
    resultado["brier_oos"] = round(float(brier), 4)
    resultado["n_teste"] = len(df_test)
    resultado["data_teste"] = {
        "inicio": datetime.utcfromtimestamp(
            int(df_test["timestamp"].iloc[0]) / 1000).strftime("%Y-%m-%d"),
        "fim": datetime.utcfromtimestamp(
            int(df_test["timestamp"].iloc[-1]) / 1000).strftime("%Y-%m-%d"),
    }

    emoji = "✅" if resultado.get("aprovado") else "❌"
    logging.info(
        f"{emoji} {symbol} | trades={resultado.get('trades')} | "
        f"win={resultado.get('winrate', 0):.1%} | PF={resultado.get('profit_factor')} | "
        f"E={resultado.get('expectativa_r', 0):+.3f}R | DD={resultado.get('max_dd_r', 0):.1f}R"
    )
    return resultado


# =====================================================================
# MAIN
# =====================================================================
async def main(simbolos: list):
    logging.info(
        f"BACKTEST PREDADOR v3.2 | tf={Config.TIMEFRAME} | barreira +/-{Config.BARRIER_PCT}% "
        f"| gate {Config.MIN_PROB_ENTRY}% | custos maker-first"
    )
    exchange = ccxt_async.bybit({"enableRateLimit": True,
                                 "options": {"defaultType": "swap"}})
    try:
        btc_bars = await fetch_historico_completo(exchange, "BTC/USDT:USDT",
                                                  Config.CANDLES_TREINAMENTO)
        relatorios = []
        for sym in simbolos:
            try:
                relatorios.append(await backtest_symbol(exchange, sym, btc_bars))
            except Exception as e:
                logging.error(f"Falha no backtest de {sym}: {e}")
                relatorios.append({"symbol": sym, "aprovado": False, "motivo": str(e)})

        # ---- Relatorio consolidado ----
        aprovados = [r for r in relatorios if r.get("aprovado")]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"backtest_{stamp}.json"
        with open(path, "w") as f:
            json.dump({"config": {
                "barrier_pct": Config.BARRIER_PCT,
                "timeframe": Config.TIMEFRAME,
                "gate": Config.MIN_PROB_ENTRY,
                "label_horizon": Config.LABEL_HORIZON_CANDLES,
            }, "resultados": [
                {k: v for k, v in r.items() if k != "trades_detalhe"} for r in relatorios
            ]}, f, indent=2, ensure_ascii=False)

        logging.info("=" * 60)
        logging.info(f"RESULTADO: {len(aprovados)}/{len(relatorios)} simbolos aprovados")
        for r in relatorios:
            if r.get("aprovado"):
                logging.info(f"  ✅ {r['symbol']}: PF={r['profit_factor']} "
                             f"E={r['expectativa_r']:+.3f}R trades={r['trades']}")
            else:
                logging.info(f"  ❌ {r['symbol']}: {r.get('motivo', 'reprovado')}")
        logging.info(f"Relatorio salvo em {path}")
        logging.info("=" * 60)

        if not aprovados:
            logging.warning(
                "NENHUM simbolo passou no gate. NAO subir para testnet. "
                "Ajustes sugeridos: BARRIER_PCT, LABEL_HORIZON_CANDLES, "
                "universo (tente so majors) ou features."
            )
    finally:
        await exchange.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("simbolos", nargs="*", default=None)
    parser.add_argument("--universo", action="store_true",
                        help="Usa top 5 do universo Config")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.universo or not args.simbolos:
        async def _pegar():
            ativos = await universe_provider.get_ativos()
            return ativos[:5]
        alvo = asyncio.run(_pegar()) if not args.simbolos else args.simbolos
    else:
        alvo = args.simbolos

    logging.info(f"Simbolos sob teste: {alvo}")
    asyncio.run(main(alvo))
