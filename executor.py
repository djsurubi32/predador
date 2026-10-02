"""
executor.py — Execucao e roteamento de ordens do Predador v3.2.

MODO TESTNET (OPERA_CONTA_REAL=False, padrao):
    URL da testnet da Bybit automatica. Dinheiro ficticio, order book e
    latencia REAIS. Nada de simulacao em SQLite.

MODO REAL (OPERA_CONTA_REAL=True):
    Bybit mainnet com as chaves do .env.

EXECUCAO MAKER-FIRST (entry):
    1. Ordem LIMIT postOnly no melhor bid (BUY) ou ask (SELL)
    2. Espera ENTRY_MAKER_TIMEOUT_SEC (20s) monitorando o fill
    3. Se nao executou: cancela e cai no TAKER (market) — sinal de 5m
       nao morre em 20 segundos, mas a entrada nao pode esperar para sempre
    4. Apos o fill: registra TP/SL na exchange (set_trading_stop)

SAIDA: market taker direto (time-stop/hard-stop sao situacoes de saida
imediata; a economia do maker nao justifica risco de nao sair).

CONTRATO COM O GERENCIADOR (packet):
    {symbol, direction, qty, entry_price, tp, sl, barrier_pct, module}
"""

import sys
import time
import asyncio
import logging
from typing import Optional, List

import requests
import ccxt
from requests.adapters import HTTPAdapter

from config import Config

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EXECUTOR] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =====================================================================
# TELEGRAM
# =====================================================================
class TelegramLogger:
    _ultimo_envio = 0.0
    _min_intervalo = 3.0  # anti-flood

    @staticmethod
    def _send_sync(message: str):
        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": Config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            }
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

    @staticmethod
    async def send(message: str):
        agora = time.time()
        if agora - TelegramLogger._ultimo_envio < TelegramLogger._min_intervalo:
            await asyncio.sleep(TelegramLogger._min_intervalo)
        TelegramLogger._ultimo_envio = time.time()
        await asyncio.to_thread(TelegramLogger._send_sync, message)


# =====================================================================
# MOTOR DE EXECUCAO BYBIT
# =====================================================================
class BybitExecutionEngine:
    def __init__(self, private_exchange=None, public_exchange=None, is_real=True):
        self.private_exchange = private_exchange
        self.public_exchange = public_exchange
        self.is_real = is_real

        self.taker_fee = Config.TAKER_FEE_PCT / 100.0
        self.maker_fee = Config.MAKER_FEE_PCT / 100.0

        self._eq_cache = 0.0
        self._eq_ts = 0.0
        self._pos_cache: List[dict] = []
        self._pos_ts = 0.0

    # ------------------------------------------------------------------
    # UTIL
    # ------------------------------------------------------------------
    def _precisao(self, symbol: str):
        mercado = self.private_exchange.market(symbol)
        return mercado["precision"]

    def _preco_valido(self, symbol: str, preco: float) -> float:
        """Ajusta o preco para o tick da Bybit e impede postOnly cruzado."""
        m = self.private_exchange.market(symbol)
        tick = float(m["precision"]["price"]) if m["precision"]["price"] else 1e-8
        # Nao deixar preco exatamente no tick = gera reject por crossing
        return round(round(preco / tick) * tick, 10)

    # ------------------------------------------------------------------
    # CONTA
    # ------------------------------------------------------------------
    async def get_equity(self) -> float:
        if not self.private_exchange:
            return float(Config.BANCA_INICIAL_USD)
        agora = time.time()
        if agora - self._eq_ts < 10.0 and self._eq_cache > 0:
            return self._eq_cache
        try:
            bal = await asyncio.to_thread(
                self.private_exchange.fetch_balance, {"accountType": "UNIFIED"}
            )
            info = (bal.get("info", {}) or {}).get("result", {}) or {}
            equity = float(
                (info.get("list") or [{}])[0].get(
                    "totalEquity", bal.get("total", {}).get("USDT", 0)
                )
            )
            if equity > 0:
                self._eq_cache = equity
                self._eq_ts = agora
            return self._eq_cache or float(Config.BANCA_INICIAL_USD)
        except Exception as e:
            logging.error(f"Equity indisponivel: {e}")
            return self._eq_cache or float(Config.BANCA_INICIAL_USD)

    def get_equity_sync_if_available(self) -> float:
        """Sync p/ gerenciador: usa cache; 0.0 indica 'buscar de outra forma'."""
        if self._eq_cache > 0 and time.time() - self._eq_ts < 60:
            return self._eq_cache
        return 0.0

    # ------------------------------------------------------------------
    # POSICOES
    # ------------------------------------------------------------------
    async def get_current_positions(self) -> List[dict]:
        if not self.private_exchange:
            return []

        agora = time.time()
        if agora - self._pos_ts < 4.0:
            return self._pos_cache

        try:
            raw = await asyncio.to_thread(self.private_exchange.fetch_positions)
            ativas = []
            for p in raw:
                size = float(p.get("contracts") or 0)
                if abs(size) <= 1e-10:
                    continue

                entry = float(p.get("entryPrice") or 0)
                mark = float(p.get("markPrice") or entry)
                gross = float(p.get("unrealizedPnl") or 0)

                # PnL liquido estimado: custo de SAIDA (taker) sobre o mark
                taxa_saida = abs(size) * mark * self.taker_fee
                taxa_entrada = abs(size) * entry * (
                    self.maker_fee  # entrada foi maker-first
                )
                net = gross - taxa_saida - taxa_entrada

                info = p.get("info", {}) or {}
                ativas.append({
                    "symbol": p.get("symbol"),
                    "side": str(p.get("side", "long")).upper(),
                    "contracts": abs(size),
                    "entryPrice": entry,
                    "markPrice": mark,
                    "grossPnl": gross,
                    "netPnl": net,
                    "stopLoss": float(info.get("stopLoss", 0) or 0),
                    "takeProfit": float(info.get("takeProfit", 0) or 0),
                })

            self._pos_cache = ativas
            self._pos_ts = agora
            return ativas
        except Exception as e:
            logging.error(f"Posicoes indisponiveis: {e}")
            return self._pos_cache

    # ------------------------------------------------------------------
    # TP/SL
    # ------------------------------------------------------------------
    async def modify_position_tp_sl(self, symbol: str,
                                    take_profit: float = None,
                                    stop_loss: float = None) -> bool:
        if not self.private_exchange:
            return False
        try:
            params = {}
            if stop_loss is not None and stop_loss > 0:
                params["stopLoss"] = str(stop_loss)
            if take_profit is not None and take_profit > 0:
                params["takeProfit"] = str(take_profit)
            if not params:
                return True
            await asyncio.to_thread(
                self.private_exchange.set_trading_stop, symbol, **params
            )
            logging.info(f"TP/SL atualizado {symbol} | TP={take_profit} SL={stop_loss}")
            return True
        except Exception as e:
            logging.error(f"Falha TP/SL {symbol}: {e}")
            return False

    # ------------------------------------------------------------------
    # FECHAMENTO
    # ------------------------------------------------------------------
    async def close_position_market(self, symbol: str, side: str,
                                    amount: float) -> bool:
        if not self.private_exchange:
            return False
        try:
            order_side = "sell" if side.upper() in ("LONG", "BUY") else "buy"
            amt = float(self.private_exchange.amount_to_precision(symbol, amount))
            if amt <= 0:
                return False
            await asyncio.to_thread(
                self.private_exchange.create_order,
                symbol, "market", order_side, amt, None, {"reduceOnly": True},
            )
            logging.info(f"FECHADO {symbol} | side={order_side} qty={amt}")
            return True
        except Exception as e:
            logging.error(f"Falha ao fechar {symbol}: {e}")
            return False

    # ------------------------------------------------------------------
    # ABERTURA MAKER-FIRST
    # ------------------------------------------------------------------
    async def _set_leverage(self, symbol: str):
        try:
            await asyncio.to_thread(
                self.private_exchange.set_leverage, 5, symbol
            )
        except Exception:
            pass  # ja setada ou erro inofensivo da Bybit

    async def abrir_posicao_maker_first(self, symbol: str, direction: str,
                                        qty: float, sl: float,
                                        tp: float) -> bool:
        """
        1. Limit postOnly no melhor bid/ask
        2. Monitora fill por ENTRY_MAKER_TIMEOUT_SEC
        3. Fallback taker se nao executar
        4. Registra TP/SL na exchange apos fill
        Retorna True se posicao aberta (de algum modo).
        """
        if not self.private_exchange:
            return False

        await self._set_leverage(symbol)

        side_u = direction.upper()
        lado = "buy" if side_u in ("BUY", "LONG") else "sell"

        try:
            ticker = await asyncio.to_thread(
                self.public_exchange.fetch_ticker, symbol
            )
            best_bid = float(ticker.get("bid", 0) or 0)
            best_ask = float(ticker.get("ask", 0) or 0)
            if best_bid <= 0 or best_ask <= 0:
                raise ValueError("book vazio")

            # postOnly: BUY no bid, SELL no ask — nunca cruza
            preco_limite = best_bid if lado == "buy" else best_ask
            preco_limite = self._preco_valido(symbol, preco_limite)

            amt = float(self.private_exchange.amount_to_precision(symbol, qty))
            if amt <= 0:
                logging.error(f"Qty invalida {symbol}: {qty}")
                return False

            logging.info(
                f"ENTRY maker {lado} {symbol} @ {preco_limite} qty={amt}"
            )
            order = await asyncio.to_thread(
                self.private_exchange.create_order,
                symbol, "limit", lado, amt, preco_limite,
                {"postOnly": True},
            )
            order_id = order.get("id")

            # ---- Monitora fill ----
            preco_filled = None
            t0 = time.time()
            while time.time() - t0 < Config.ENTRY_MAKER_TIMEOUT_SEC:
                await asyncio.sleep(1.0)
                try:
                    o = await asyncio.to_thread(
                        self.private_exchange.fetch_order, order_id, symbol
                    )
                except Exception:
                    continue
                filled = float(o.get("filled", 0) or 0)
                if filled >= amt * 0.999:
                    preco_filled = float(o.get("average") or preco_limite)
                    logging.info(
                        f"MAKER FILLED {symbol} @ {preco_filled:.6f} "
                        f"(+{Config.MAKER_FEE_PCT}% fee)"
                    )
                    break

            if preco_filled is None:
                # Fallback taker
                try:
                    await asyncio.to_thread(
                        self.private_exchange.cancel_order, order_id, symbol
                    )
                except Exception:
                    pass
                await asyncio.sleep(0.3)
                o = await asyncio.to_thread(
                    self.private_exchange.fetch_order, order_id, symbol
                )
                filled_maker = float(o.get("filled", 0) or 0)
                restante = amt - filled_maker

                if restante > 1e-8:
                    logging.info(
                        f"Maker nao executou — fallback TAKER {symbol} "
                        f"(restante {restante})"
                    )
                    await asyncio.to_thread(
                        self.private_exchange.create_order,
                        symbol, "market", lado, restante, None,
                    )

                posicoes = await self.get_current_positions()
                achou = next(
                    (p for p in posicoes if p.get("symbol") == symbol
                     and float(p.get("contracts", 0)) > 1e-8),
                    None,
                )
                if not achou:
                    logging.error(f"Fallback falhou — sem posicao em {symbol}")
                    return False

            # ---- Registra TP/SL na exchange ----
            await asyncio.sleep(0.3)
            ok = await self.modify_position_tp_sl(symbol, tp, sl)
            if not ok:
                await TelegramLogger.send(
                    f"⚠️ {symbol} aberta SEM TP/SL na exchange — "
                    f"corrigir manualmente (SL={sl:.6f} TP={tp:.6f})"
                )
            return True

        except Exception as e:
            logging.error(f"Falha na abertura {symbol}: {e}")
            return False


# =====================================================================
# FACHADA — o que o gerenciador instancia
# =====================================================================
class EngineExecutor:
    def __init__(self):
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        is_real = Config.OPERA_CONTA_REAL

        # ---- Publica (tickers/orderbook) ----
        self.public_exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
            "session": session,
        })
        if not is_real:
            self.public_exchange.set_sandbox_mode(True)

        # ---- Privada ----
        self.private_exchange = None
        if Config.BYBIT_API_KEY and Config.BYBIT_SECRET:
            try:
                self.private_exchange = ccxt.bybit({
                    "apiKey": Config.BYBIT_API_KEY.strip('" ').strip(),
                    "secret": Config.BYBIT_SECRET.strip('" ').strip(),
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap", "recvWindow": 10000},
                })
                if not is_real:
                    self.private_exchange.set_sandbox_mode(True)
            except Exception as e:
                logging.error(f"Credenciais Bybit invalidas: {e}")

        if is_real and not self.private_exchange:
            raise RuntimeError(
                "OPERA_CONTA_REAL=True sem chaves validas. "
                "Corrija o .env ou desligue o modo real."
            )

        self.execution = BybitExecutionEngine(
            private_exchange=self.private_exchange,
            public_exchange=self.public_exchange,
            is_real=is_real,
        )

    # ------------------------------------------------------------------
    # ROTEADOR — chamado pelo gerenciador
    # ------------------------------------------------------------------
    async def order_router_inbound(self, packet: dict) -> bool:
        symbol = packet["symbol"]
        direction = packet["direction"]
        qty = float(packet["qty"])
        tp = float(packet.get("tp") or 0.0)
        sl = float(packet.get("sl") or 0.0)

        if qty <= 0 or not self.private_exchange:
            return False

        # Posicao ja existe? Nao duplica
        posicoes = await self.execution.get_current_positions()
        if any(p.get("symbol") == symbol for p in posicoes):
            logging.warning(f"{symbol}: posicao ja existe — roteador recusou.")
            return False

        return await self.execution.abrir_posicao_maker_first(
            symbol, direction, qty, sl, tp
        )
