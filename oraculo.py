"""
oraculo.py — Filtro institucional de fluxo do Predador v3.2.

PAPEL: ultima validacao ANTES do gerenciador abrir posicao.
    Aprova somente se o fluxo (L2 + CVD) NAO estiver contra a direcao.

FONTE: Binance Futures (API publica, gratuita, sem chave).
    Binance tem o book mais liquido do mundo — e o melhor proxy
    de agressao institucional disponivel de graca.

REGRAS:
    BUY  : L2 >= -ORACULO_L2_TOL e CVD >= -ORACULO_CVD_TOL (nao contra)
    SELL : L2 <= +ORACULO_L2_TOL e CVD <= +ORACULO_CVD_TOL (nao contra)
    Bonus "FORTE" quando ambos a favor — log informativo.

FALHA DE API = BYPASS (aprovado=True).
    O Oraculo filtra; nunca paralisa o bot. Se a Binance cair,
    o sistema segue operando com o gate de 65% sozinho.

CVD: proxy de agressao nos ultimos 10 candles de 5m (~50 min):
    volume de velas de alta menos volume de velas de baixa,
    normalizado pelo volume total.
"""

import logging
import asyncio

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ORACULO] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CVD_LOOKBACK = 10  # velas de 5m (~50 minutos)


class OraculoBinance:
    def __init__(self):
        self.exchange = ccxt_async.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        self._mercados_carregados = False

        # Tolerancias vindas do Config (nao hardcoded)
        self.l2_tol = Config.ORACULO_L2_TOL      # 0.02
        self.cvd_tol = Config.ORACULO_CVD_TOL    # 0.03

        # Limiar "forte" (informativo): ambos claramente a favor
        self.l2_forte = 0.08
        self.cvd_forte = 0.10

    async def _garantir_mercados(self):
        if not self._mercados_carregados:
            await self.exchange.load_markets()
            self._mercados_carregados = True

    async def fechar_conexoes(self):
        try:
            await self.exchange.close()
        except Exception:
            pass

    @staticmethod
    def traduzir_simbolo(symbol_bybit: str) -> str:
        """BTC/USDT:USDT -> BTC/USDT"""
        return symbol_bybit.split(":")[0]

    # ------------------------------------------------------------------
    # L2: imbalance financeiro top-50 do book
    # ------------------------------------------------------------------
    async def analisar_imbalance_l2(self, symbol: str) -> float:
        """
        +1.0 = so compradores no book | -1.0 = so vendedores | 0 = neutro
        """
        try:
            ob = await asyncio.wait_for(
                self.exchange.fetch_order_book(symbol, limit=50),
                timeout=4.0,
            )
            bids = np.array(ob.get("bids") or [], dtype=float).reshape(-1, 2)
            asks = np.array(ob.get("asks") or [], dtype=float).reshape(-1, 2)

            vol_bids = float((bids[:, 0] * bids[:, 1]).sum()) if len(bids) else 0.0
            vol_asks = float((asks[:, 0] * asks[:, 1]).sum()) if len(asks) else 0.0
            return float((vol_bids - vol_asks) / (vol_bids + vol_asks + 1e-9))
        except Exception as e:
            logging.debug(f"L2 falhou {symbol}: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # CVD: agressao nos ultimos N candles (proxy de delta de volume)
    # ------------------------------------------------------------------
    async def analisar_cvd(self, symbol: str) -> float:
        """
        Positivo = agressao compradora dominante | Negativo = vendedora
        """
        try:
            ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(symbol, Config.TIMEFRAME, limit=CVD_LOOKBACK),
                timeout=4.0,
            )
            if not ohlcv or len(ohlcv) < 5:
                return 0.0

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            direcao = np.where(df["close"] > df["open"], 1.0,
                               np.where(df["close"] < df["open"], -1.0, 0.0))
            delta = float((df["volume"] * direcao).sum())
            total = float(df["volume"].sum())
            return float(delta / (total + 1e-9))
        except Exception as e:
            logging.debug(f"CVD falhou {symbol}: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # VALIDACAO PRINCIPAL
    # ------------------------------------------------------------------
    async def validar_sinal_institucional(self, symbol_bybit: str,
                                          direcao: str) -> tuple[bool, str]:
        """
        Retorna (aprovado, motivo).
        Falha de API -> bypass (True, 'indisponivel').
        """
        try:
            await self._garantir_mercados()
            symbol = self.traduzir_simbolo(symbol_bybit)

            l2, cvd = await asyncio.gather(
                self.analisar_imbalance_l2(symbol),
                self.analisar_cvd(symbol),
            )

            tag = f"L2={l2:+.3f} CVD={cvd:+.3f}"
            lado = direcao.upper()

            if lado in ("BUY", "LONG"):
                ok_l2 = l2 >= -self.l2_tol
                ok_cvd = cvd >= -self.cvd_tol
                if ok_l2 and ok_cvd:
                    forte = l2 >= self.l2_forte and cvd >= self.cvd_forte
                    return True, f"{'FORTE' if forte else 'OK'} BUY {tag}"
                return False, f"BLOQUEADO BUY {tag}"

            if lado in ("SELL", "SHORT"):
                ok_l2 = l2 <= self.l2_tol
                ok_cvd = cvd <= self.cvd_tol
                if ok_l2 and ok_cvd:
                    forte = l2 <= -self.l2_forte and cvd <= -self.cvd_forte
                    return True, f"{'FORTE' if forte else 'OK'} SELL {tag}"
                return False, f"BLOQUEADO SELL {tag}"

            return False, f"Direcao desconhecida: {direcao}"

        except Exception as e:
            logging.error(f"Oraculo indisponivel {symbol_bybit}: {e}")
            return True, "Oraculo indisponivel - bypass"


# =====================================================================
# TESTE ISOLADO
# =====================================================================
if __name__ == "__main__":
    async def _teste():
        o = OraculoBinance()
        casos = [
            ("BTC/USDT:USDT", "BUY"),
            ("ETH/USDT:USDT", "SELL"),
            ("SOL/USDT:USDT", "BUY"),
        ]
        for sym, side in casos:
            ok, msg = await o.validar_sinal_institucional(sym, side)
            print(f"{side:5} {sym}: aprovado={ok} | {msg}")
        await o.fechar_conexoes()

    asyncio.run(_teste())
