import sys
import time
import asyncio
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import aiosqlite
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

from config import Config, BASE_DIR, universe_provider
from features import (
    adicionar_features,
    aplicar_regime_hmm,
    FEATURE_NAMES,
    OHLCV_LIMIT_INFERENCIA,
)
# Necessario para deserializacao transparente do pipeline via joblib
from treinador import CalibratedStackingEnsemble, DummyHMM  # noqa: F401

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ANALISADOR] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Precisao minima no gate (medida pelo treinador no bloco de calibracao).
# Breakeven com custos maker ~60%; abaixo de 62% nao ha edge apos custos.
MIN_PRECISAO_GATE = 0.62

TF_SECONDS = {"5m": 300, "1m": 60, "15m": 900}


class RadarCore:
    def __init__(self):
        self.exchange = ccxt_async.bybit(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        self.modelos: Dict[str, dict] = {}
        self.modelos_dir = Path(Config.MODELS_DIR)
        self.db_name = str(BASE_DIR / Config.DB_NAME)

        self.ultimo_ts: Dict[str, int] = {}   # simbolo -> ts da ultima vela processada
        self.last_reload = 0.0
        self.reload_every = 300
        self.btc_symbol = "BTC/USDT:USDT"

    # ------------------------------------------------------------------
    # MODELOS
    # ------------------------------------------------------------------
    def carregar_modelos(self):
        self.modelos.clear()
        if not self.modelos_dir.exists():
            logging.warning(f"Pasta de modelos inexistente: {self.modelos_dir}")
            return

        for path in sorted(self.modelos_dir.glob("*.pkl")):
            try:
                payload = joblib.load(path)
                if not isinstance(payload, dict) or "modelo" not in payload:
                    logging.warning(f"{path.name}: payload sem 'modelo'. Ignorado.")
                    continue

                symbol = payload.get("symbol") or path.stem.replace("_", "/")

                # Coerencia de contrato: barreira do modelo == barreira do Config
                barrier_modelo = float(payload.get("barrier_pct", 0.0))
                if barrier_modelo and abs(barrier_modelo - Config.BARRIER_PCT) > 1e-9:
                    logging.warning(
                        f"{path.name}: barreira {barrier_modelo}% != "
                        f"Config {Config.BARRIER_PCT}%. Modelo descartado."
                    )
                    continue

                # Gate de qualidade: so usa modelo comprovado no gate
                metrics = payload.get("metrics", {}) or {}
                prec = metrics.get("precisao_no_gate_65")
                if prec is not None and prec < MIN_PRECISAO_GATE:
                    logging.info(
                        f"{symbol}: precisao no gate {prec:.1%} < "
                        f"{MIN_PRECISAO_GATE:.0%}. Modelo nao operacional."
                    )
                    continue
                if prec is None:
                    logging.info(f"{symbol}: gate sem amostra suficiente no treino. Cautela.")

                self.modelos[symbol] = payload
            except Exception as e:
                logging.error(f"Falha ao carregar {path.name}: {e}")

        logging.info(f"Modelos operacionais carregados: {len(self.modelos)}")

    # ------------------------------------------------------------------
    # DADOS — mesmo contrato bruto do treinador
    # ------------------------------------------------------------------
    async def buscar_bruto(self, symbol: str, limit: int) -> Optional[list]:
        """OHLCV + OI + funding alinhados (forward-fill), como no treinador."""
        try:
            ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(symbol, Config.TIMEFRAME, limit=limit),
                timeout=10.0,
            )
            if not ohlcv or len(ohlcv) < limit // 2:
                return None

            oi_map = {}
            try:
                oi_data = await self.exchange.fetch_open_interest_history(
                    symbol, Config.TIMEFRAME, limit=limit
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
            except Exception:
                pass

            fr_map = {}
            try:
                fr_data = await self.exchange.fetch_funding_rate_history(
                    symbol, limit=min(200, limit)
                )
                for item in fr_data:
                    fr_map[int(item.get("timestamp", 0))] = float(
                        item.get("fundingRate", 0) or 0
                    )
            except Exception:
                pass

            merged, last_oi, last_fr = [], 0.0, 0.0
            for bar in ohlcv:
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
            logging.debug(f"Dados {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------
    async def analisar_symbol(self, symbol: str) -> Optional[dict]:
        payload = self.modelos.get(symbol)
        if payload is None:
            return None

        bars = await self.buscar_bruto(symbol, OHLCV_LIMIT_INFERENCIA)
        if bars is None:
            return None

        # ---- Benchmark BTC (mesmo papel do treino) ----
        btc_bars = await self.buscar_bruto(self.btc_symbol, OHLCV_LIMIT_INFERENCIA)
        btc_df = None
        if btc_bars:
            btc_df = pd.DataFrame(
                btc_bars,
                columns=["timestamp", "open", "high", "low", "close",
                         "volume", "open_interest", "funding_rate"],
            )[["timestamp", "close"]]

        df = pd.DataFrame(
            bars,
            columns=["timestamp", "open", "high", "low", "close",
                     "volume", "open_interest", "funding_rate"],
        )

        try:
            df_feat = adicionar_features(df, btc_df)
        except ValueError as e:
            logging.debug(f"Features {symbol}: {e}")
            return None

        # ---- Regime HMM (modelo salvo no pkl) ----
        try:
            df_feat["hmm_regime"] = aplicar_regime_hmm(
                df_feat, payload["hmm"]
            ).values
        except Exception:
            df_feat["hmm_regime"] = 0

        feature_cols = payload.get("feature_names") or (FEATURE_NAMES + ["hmm_regime"])
        X = pd.DataFrame(index=df_feat.index)
        for f in feature_cols:
            X[f] = df_feat[f] if f in df_feat.columns else 0.0

        row = X.iloc[[-1]]

        try:
            modelo = payload["modelo"]
            if not hasattr(modelo, "predict_proba"):
                return None
            proba = modelo.predict_proba(row)[0]
            p_up = float(proba[1]) if len(proba) >= 2 else float(proba[0])
        except Exception as e:
            logging.debug(f"Predict {symbol}: {e}")
            return None

        price = float(df_feat["close"].iloc[-1])
        bb = float(df_feat["bb_pos"].iloc[-1])

        gate = Config.MIN_PROB_ENTRY / 100.0
        direction, prob = None, 0.0
        if p_up >= gate:
            direction, prob = "BUY", p_up
        elif (1.0 - p_up) >= gate:
            direction, prob = "SELL", 1.0 - p_up
        else:
            return None

        # Vetos de localizacao (BB): nao compra no topo, nao vende no fundo
        if direction == "BUY" and bb >= Config.BB_POS_TOP_VETO:
            return None
        if direction == "SELL" and bb <= Config.BB_POS_BOTTOM_VETO:
            return None

        score = float(np.clip((prob - 0.5) / 0.5 * 10.0, 0.0, 10.0))

        return {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "prob": prob * 100.0,
            "score": score,
            "bb_pos": bb,
            "p_up": p_up,
        }

    # ------------------------------------------------------------------
    # PERSISTENCIA — contrato com o gerenciador
    # ------------------------------------------------------------------
    async def emitir_sinal(self, s: dict):
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=5000;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS elite_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    price REAL NOT NULL,
                    prob REAL NOT NULL,
                    score REAL NOT NULL,
                    barrier_pct REAL NOT NULL,
                    bb_pos REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_elite_ts ON elite_signals(timestamp)"
            )
            # Um sinal por simbolo: substitui o anterior
            await conn.execute(
                "DELETE FROM elite_signals WHERE symbol = ?", (s["symbol"],)
            )
            await conn.execute(
                """
                INSERT INTO elite_signals
                (symbol, direction, price, prob, score, barrier_pct, bb_pos, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s["symbol"],
                    s["direction"],
                    s["price"],
                    s["prob"],
                    s["score"],
                    Config.BARRIER_PCT,
                    s["bb_pos"],
                    time.time(),
                ),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # LOOP PRINCIPAL
    # ------------------------------------------------------------------
    async def scan_market(self):
        logging.info(
            f"RADAR ONLINE | tf={Config.TIMEFRAME} | ciclo={Config.CICLO_MS}ms | "
            f"gate={Config.MIN_PROB_ENTRY}% | barreira +/-{Config.BARRIER_PCT}%"
        )

        # CORREÇÃO AQUI: Carrega os mercados antes do loop e faz fallback se o DNS falhar
        try:
            await self.exchange.load_markets()
        except Exception:
            self.exchange.hostname = "bytick.com"
            await self.exchange.load_markets()

        self.carregar_modelos()
        self.last_reload = time.time()

        try:
            while True:
                t0 = time.time()
                try:
                    if time.time() - self.last_reload > self.reload_every:
                        self.carregar_modelos()
                        self.last_reload = time.time()

                    ativos = await universe_provider.get_ativos()
                    if not ativos:
                        await asyncio.sleep(Config.CICLO_MS / 1000.0)
                        continue

                    emitidos = 0
                    for symbol in ativos:
                        if symbol not in self.modelos:
                            continue

                        try:
                            # Sobe o cálculo pesado só quando fecha vela nova
                            probe = await asyncio.wait_for(
                                self.exchange.fetch_ohlcv(
                                    symbol, Config.TIMEFRAME, limit=2
                                ),
                                timeout=5.0,
                            )
                            if not probe:
                                continue
                            ts_nova = int(probe[-1][0])
                            if self.ultimo_ts.get(symbol) == ts_nova:
                                continue
                            # Espera a vela fechar de fato (+2s de folga)
                            tf_sec = TF_SECONDS.get(Config.TIMEFRAME, 300)
                            idade = time.time() - (ts_nova / 1000.0)
                            if idade < tf_sec + 2:
                                continue

                            self.ultimo_ts[symbol] = ts_nova
                            res = await self.analisar_symbol(symbol)
                            if res is None:
                                continue

                            await self.emitir_sinal(res)
                            emitidos += 1
                            logging.info(
                                f"SINAL {res['direction']} {res['symbol']} | "
                                f"P={res['prob']:.1f}% score={res['score']:.1f} "
                                f"px={res['price']:.6f} bb={res['bb_pos']:.2f}"
                            )
                        except Exception as e:
                            logging.debug(f"Erro {symbol}: {e}")

                        await asyncio.sleep(0.05)

                    if emitidos:
                        logging.info(f"Ciclo: {emitidos} sinais emitidos")

                except Exception as e:
                    logging.error(f"Erro no scan: {e}")

                elapsed = time.time() - t0
                await asyncio.sleep(max(0.5, Config.CICLO_MS / 1000.0 - elapsed))
        finally:
            try:
                await self.exchange.close()
            except Exception:
                pass


async def main():
    radar = RadarCore()
    await radar.scan_market()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Analisador encerrado.")
