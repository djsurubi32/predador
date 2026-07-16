import sys
import os
import asyncio
import math
import logging
import time
import sqlite3
import hashlib
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
import ccxt
from config import Config

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EXECUTOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class TelegramLogger:
    @staticmethod
    def _send_sync(message: str):
        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            logging.warning("⚠️ Token ou Chat ID do Telegram não configurados no arquivo .env")
            return
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
            response = requests.post(url, json=payload, timeout=5)

            if response.status_code != 200:
                logging.error(f"❌ Falha no envio do Telegram (Status {response.status_code}): {response.text}")
        except Exception as e:
            logging.error(f"❌ Erro crítico de rede ao conectar com a API do Telegram: {e}")

    @staticmethod
    async def send(message: str):
        await asyncio.to_thread(TelegramLogger._send_sync, message)

class BybitExecutionEngine:
    def __init__(self, private_exchange=None, public_exchange=None, is_real_mode=True):
        self.private_exchange = private_exchange
        self.public_exchange = public_exchange
        self.is_real_mode = is_real_mode
        self.last_equity_fetch = 0
        self.last_positions_fetch = 0
        self.equity_cache = Config.BANCA_DEMO_INICIAL
        self.positions_cache = []

    def init_markets_sync(self):
        if self.private_exchange:
            try:
                self.private_exchange.load_markets()
                logging.info("Mecanismo de roteamento de ordens verificado com sucesso.")
            except Exception as e:
                logging.error(f"❌ Erro ao inicializar mercados na corretora: {e}")

    async def set_leverage(self, symbol: str, leverage: int = 50):
        if not self.private_exchange or not self.is_real_mode: return True
        try:
            await asyncio.to_thread(self.private_exchange.set_leverage, leverage, symbol)
            return True
        except Exception:
            return True

    async def get_equity(self):
        if not self.private_exchange or not self.is_real_mode: return Config.BANCA_DEMO_INICIAL
        now = time.time()
        if now - self.last_equity_fetch < 3.0: return self.equity_cache
        try:
            balance_data = await asyncio.to_thread(self.private_exchange.fetch_balance, {'accountType': 'UNIFIED'})
            raw_info = balance_data.get('info', {}).get('result', {}).get('list', [{}])[0]
            equity = float(raw_info.get('totalEquity', balance_data.get('total', {}).get('USDT', 0)))
            if equity > 0:
                self.equity_cache = equity
                self.last_equity_fetch = now
            return self.equity_cache
        except Exception:
            return self.equity_cache

    async def get_current_positions(self, db_reference=None):
        if self.is_real_mode:
            if not self.private_exchange: return []
            now = time.time()
            if now - self.last_positions_fetch < 4.0: return self.positions_cache
            try:
                positions = await asyncio.to_thread(self.private_exchange.fetch_positions)
                active = []
                for p in positions:
                    contracts = p.get('contracts') or p.get('positionAmt')
                    if contracts is None: continue
                    size = float(contracts)

                    if abs(size) > 0.00001:
                        entry_price = float(p.get('entryPrice') or 0)
                        mark_price = float(p.get('markPrice') or entry_price)
                        unrealized_gross = float(p.get('unrealizedPnl') or 0)

                        volume_entrada = abs(size) * entry_price
                        volume_saida = abs(size) * mark_price
                        taxa_estimada = (volume_entrada * 0.00055) + (volume_saida * 0.00055)
                        net_pnl = unrealized_gross - taxa_estimada

                        active.append({
                            'symbol': p.get('symbol'),
                            'side': str(p.get('side', 'long')).upper(),
                            'contracts': abs(size),
                            'entryPrice': entry_price,
                            'grossPnl': unrealized_gross,
                            'netPnl': net_pnl,
                            'estimatedFee': taxa_estimada
                        })
                self.positions_cache = active
                self.last_positions_fetch = now
                return active
            except Exception:
                return self.positions_cache
        else:
            if db_reference is None: return []
            try:
                open_trades = await db_reference.get_all_open_trades()
                if not open_trades: return []

                tickers = await asyncio.to_thread(self.public_exchange.fetch_tickers)
                active = []

                for t in open_trades:
                    symbol, side, entry, qty, open_time = t[0], t[1], float(t[2]), float(t[3]), float(t[4])
                    ticker = tickers.get(symbol)
                    if not ticker: continue

                    current_price = float(ticker['last'] or ticker['close'] or entry)

                    if side.upper() == 'BUY':
                        unrealized_gross = (current_price - entry) * qty
                    else:
                        unrealized_gross = (entry - current_price) * qty

                    volume_entrada = qty * entry
                    volume_saida = qty * current_price
                    taxa_estimada = (volume_entrada * 0.00055) + (volume_saida * 0.00055)
                    net_pnl = unrealized_gross - taxa_estimada

                    active.append({
                        'symbol': symbol,
                        'side': 'LONG' if side.upper() == 'BUY' else 'SHORT',
                        'contracts': qty,
                        'entryPrice': entry,
                        'grossPnl': unrealized_gross,
                        'netPnl': net_pnl,
                        'estimatedFee': taxa_estimada
                    })
                return active
            except Exception:
                return []

    async def close_position_market(self, symbol: str, side: str, amount: float):
        if not self.private_exchange or not self.is_real_mode: return True
        try:
            order_side = 'sell' if side.upper() == 'BUY' else 'buy'
            amount_str = self.private_exchange.amount_to_precision(symbol, amount)
            await asyncio.to_thread(self.private_exchange.create_order, symbol, 'market', order_side, float(amount_str), None, {'reduceOnly': True})
            return True
        except Exception as e:
            logging.error(f"Erro ao fechar posição a mercado {symbol}: {e}")
            return False

    async def close_position_limit_chase(self, symbol: str, side: str, amount: float, max_retries: int = 5):
        if not self.private_exchange or not self.is_real_mode: return True
        order_side = 'sell' if side.upper() == 'BUY' else 'buy'
        remaining_amount = amount

        for attempt in range(max_retries):
            try:
                ticker = await asyncio.to_thread(self.public_exchange.fetch_ticker, symbol)
                limit_price = float(ticker['ask']) if order_side == 'sell' else float(ticker['bid'])
                amount_str = self.private_exchange.amount_to_precision(symbol, remaining_amount)
                price_str = self.private_exchange.price_to_precision(symbol, limit_price)

                order = await asyncio.to_thread(self.private_exchange.create_order, symbol, 'limit', order_side, float(amount_str), float(price_str), {'reduceOnly': True, 'postOnly': True})
                await asyncio.sleep(3)

                fetched_order = await asyncio.to_thread(self.private_exchange.fetch_order, order['id'], symbol, params={'acknowledged': True})
                if fetched_order.get('status') == 'closed': return True
                else:
                    await asyncio.to_thread(self.private_exchange.cancel_order, order['id'], symbol)
                    await asyncio.sleep(0.5)
                    fetched_order_after = await asyncio.to_thread(self.private_exchange.fetch_order, order['id'], symbol, params={'acknowledged': True})
                    filled = float(fetched_order_after.get('filled', 0.0))
                    remaining_amount = amount - filled
                    if remaining_amount <= 0.00001: return True
            except Exception:
                await asyncio.sleep(1)

        if remaining_amount > 0.00001:
            await self.close_position_market(symbol, side, remaining_amount)
        return True

    async def open_position_limit(self, symbol: str, side: str, amount: float, limit_price: float):
        if not self.private_exchange or not self.is_real_mode: return {"status": "simulated"}
        try:
            await self.set_leverage(symbol, Config.ALAVANCAGEM)
            order_side = 'buy' if side.upper() == 'BUY' else 'sell'
            amount_str = self.private_exchange.amount_to_precision(symbol, amount)
            price_str = self.private_exchange.price_to_precision(symbol, limit_price)
            order = await asyncio.to_thread(self.private_exchange.create_order, symbol, 'limit', order_side, float(amount_str), float(price_str))
            return order
        except Exception as e:
            logging.error(f"Erro ao abrir posição Limit em {symbol}: {e}")
            raise

class Database:
    def __init__(self, db_name="predador_v31.db"):
        self.db_name = db_name
        self._create_tables()

    def _get_connection(self):
        # 🛡️ FIX COORDENAÇÃO WAL: Garante que todas as conexões SQLite ativem o WAL imediatamente ao abrir
        conn = sqlite3.connect(self.db_name, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, entry REAL,
                sl REAL, tp REAL, qty REAL, force INTEGER, open_time REAL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS elite_signals (
                symbol TEXT PRIMARY KEY, direction TEXT, price REAL, prob REAL,
                score REAL, atr REAL, sma_atr REAL, funding REAL, reasoning TEXT,
                timestamp REAL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
                pnl REAL, outcome TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()

    def _get_elite_signals_sync(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning FROM elite_signals')
            rows = cursor.fetchall()
            return [{'symbol': r[0], 'direction': r[1], 'price': r[2], 'prob': r[3], 'score': r[4], 'current_atr': r[5], 'sma_atr': r[6], 'funding': r[7], 'reasoning': r[8]} for r in rows]

    async def get_elite_signals(self):
        return await asyncio.to_thread(self._get_elite_signals_sync)

    def _add_trade_sync(self, symbol, side, entry, sl, tp, qty, force):
        trade_id = hashlib.sha256(f"{symbol}{side}{time.time():.4f}".encode()).hexdigest()[:16]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO trades (trade_id, symbol, side, entry, sl, tp, qty, force, open_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (trade_id, symbol, side, entry, sl, tp, qty, force, time.time()))
            conn.commit()
        return trade_id

    async def add_trade(self, symbol, side, entry, sl, tp, qty, force):
        return await asyncio.to_thread(self._add_trade_sync, symbol, side, entry, sl, tp, qty, force)

    def _get_all_open_trades_sync(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT symbol, side, entry, qty, open_time FROM trades')
            return cursor.fetchall()

    async def get_all_open_trades(self):
        return await asyncio.to_thread(self._get_all_open_trades_sync)

    def _remove_trade_sync(self, symbol):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM trades WHERE symbol = ?', (symbol,))
            conn.commit()

    async def remove_trade(self, symbol):
        await asyncio.to_thread(self._remove_trade_sync, symbol)

    def _add_history_sync(self, symbol, side, pnl, outcome):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO history (symbol, side, pnl, outcome) VALUES (?, ?, ?, ?)', (symbol, side, pnl, outcome))
            conn.commit()

    async def add_history(self, symbol, side, pnl, outcome):
        await asyncio.to_thread(self._add_history_sync, symbol, side, pnl, outcome)

class EngineExecutor:
    def __init__(self):
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        self.public_exchange = ccxt.bybit({'enableRateLimit': True, 'rateLimit': 50, 'options': {'defaultType': 'swap'}, 'session': session})
        self.private_exchange = None
        if Config.BYBIT_API_KEY and Config.BYBIT_API_KEY != "SUA_API_KEY_BYBIT" and Config.BYBIT_API_KEY.strip() != "":
            try:
                self.private_exchange = ccxt.bybit({'apiKey': Config.BYBIT_API_KEY, 'secret': Config.BYBIT_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap', 'recvWindow': 10000}, 'session': session})
            except Exception as e:
                logging.error(f"Erro nas credenciais: {e}")

        self.execution = BybitExecutionEngine(private_exchange=self.private_exchange, public_exchange=self.public_exchange, is_real_mode=Config.OPERA_CONTA_REAL)
        self.db = Database()

        self.max_basket_pnl = 0.0
        self.min_basket_pnl = 0.0
        self.last_global_status_time = time.time()
        self.last_empty_heartbeat = time.time()
        self.last_summary_time = time.time()
        self.cooldown_memoria = {}
        self.simulated_banca = float(Config.BANCA_DEMO_INICIAL)

    def route_and_calculate_strategy(self, opp):
        score = float(opp['score'])
        prob = float(opp['prob'])

        reasoning_clean = opp['reasoning'].replace(" ", "")
        is_fluxo_maximo = "Fluxo:3.0" in reasoning_clean
        is_espaco_expandido = "ML_Espaço:3.0" in reasoning_clean

        if score >= 9.0 and prob >= 75.0:
            return {
                "vertente": "QUALIDADE EXTREMA (SNIPER)",
                "lote_tipo": "Lote Sniper",
                "invest_amount": 6.0,
                "tp_factor": 1.020,
                "sl_factor": 0.990
            }
        elif is_fluxo_maximo:
            return {
                "vertente": "SCALPING DE MOMENTUM",
                "lote_tipo": "Lote Padrão",
                "invest_amount": 4.0,
                "tp_factor": 1.008,
                "sl_factor": 0.995
            }
        elif is_espaco_expandido:
            return {
                "vertente": "DAY TRADE DE EXPANSÃO",
                "lote_tipo": "Lote Leve",
                "invest_amount": 3.0,
                "tp_factor": 1.040,
                "sl_factor": 0.980
            }

        return {
            "vertente": "PADRÃO ADAPTATIVO",
            "lote_tipo": "Lote de Teste",
            "invest_amount": 2.0,
            "tp_factor": 1.012,
            "sl_factor": 0.988
        }

    async def check_btc_trend_1h(self) -> str:
        try:
            candles = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, 'BTC/USDT:USDT', '1h', limit=20)
            if not candles or len(candles) < 20:
                return "NEUTRAL"
            closes = [float(c[4]) for c in candles]
            sma = sum(closes) / len(closes)
            current_price = closes[-1]
            if current_price > sma: return "BULLISH"
            elif current_price < sma: return "BEARISH"
            return "NEUTRAL"
        except Exception as e:
            logging.error(f"Erro ao checar tendência macro do BTC (1h): {e}")
            return "NEUTRAL"

    async def check_asset_trend_4h(self, symbol: str) -> str:
        try:
            candles = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, symbol, '4h', limit=20)
            if not candles or len(candles) < 20:
                return "NEUTRAL"
            closes = [float(c[4]) for c in candles]
            sma = sum(closes) / len(closes)
            current_price = closes[-1]
            if current_price > sma: return "BULLISH"
            elif current_price < sma: return "BEARISH"
            return "NEUTRAL"
        except Exception as e:
            logging.error(f"Erro ao checar timeframe macro 4h para {symbol}: {e}")
            return "NEUTRAL"

    # 🛡️ NOVO REQUISITO INSTITUCIONAL: Validação estrita de timeframe de 1h da moeda
    async def check_asset_trend_1h(self, symbol: str) -> str:
        try:
            candles = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, symbol, '1h', limit=20)
            if not candles or len(candles) < 20:
                return "NEUTRAL"
            closes = [float(c[4]) for c in candles]
            sma = sum(closes) / len(closes)
            current_price = closes[-1]
            if current_price > sma: return "BULLISH"
            elif current_price < sma: return "BEARISH"
            return "NEUTRAL"
        except Exception as e:
            logging.error(f"Erro ao checar timeframe macro 1h para {symbol}: {e}")
            return "NEUTRAL"

    async def check_open_interest_healthy(self, symbol: str) -> bool:
        try:
            oi_data = await asyncio.to_thread(self.public_exchange.fetch_open_interest, symbol)
            if oi_data and len(oi_data) > 0:
                oi_value = float(oi_data[0].get('openInterestAmount', 0) or 0)
                return oi_value > 0
            return True
        except Exception:
            return True

    async def reconcile_positions(self):
        if not Config.OPERA_CONTA_REAL: return
        try:
            real_positions = await self.execution.get_current_positions()
            real_symbols = {p['symbol'] for p in real_positions}
            open_trades = await self.db.get_all_open_trades()

            for trade in open_trades:
                symbol = trade[0]
                open_time = trade[4]
                if symbol not in real_symbols:
                    if time.time() - open_time < 30.0: continue
                    if self.private_exchange:
                        orders = await asyncio.to_thread(self.private_exchange.fetch_open_orders, symbol)
                        if len(orders) == 0:
                            await self.db.remove_trade(symbol)
                    else:
                        await self.db.remove_trade(symbol)
        except Exception:
            pass

    async def cancel_old_pending_orders(self):
        if not Config.OPERA_CONTA_REAL or not self.private_exchange: return
        try:
            open_trades = await self.db.get_all_open_trades()
            for trade in open_trades:
                symbol = trade[0]
                try:
                    orders = await asyncio.to_thread(self.private_exchange.fetch_open_orders, symbol)
                    now = time.time()
                    for order in orders:
                        order_time = order['timestamp'] / 1000.0
                        if now - order_time >= (Config.MAX_PENDING_ORDER_MINUTES * 60):
                            await asyncio.to_thread(self.private_exchange.cancel_order, order['id'], symbol)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        except Exception:
            pass

    async def manage_basket(self):
        positions = await self.execution.get_current_positions(self.db)
        now = time.time()

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            self.last_summary_time = now
            if now - self.last_empty_heartbeat > 60:
                logging.info("⚖️ Balde vazio. Aguardando oportunidades das vertentes...")
                self.last_empty_heartbeat = now
            return

        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        total_margin = sum((float(p.get('contracts', 0)) * float(p.get('entryPrice', 0))) / Config.ALAVANCAGEM for p in positions)

        if total_margin <= 0: return

        # 🛡️ AJUSTE MATEMÁTICO INSTITUCIONAL: Tratamento dinâmico de porcentagem (valores inteiros para decimais reais)
        f_target = Config.BASKET_TARGET_PCT / 100.0 if Config.BASKET_TARGET_PCT >= 0.05 else Config.BASKET_TARGET_PCT
        f_pullback = Config.BASKET_TRAILING_PULLBACK_PCT / 100.0 if Config.BASKET_TRAILING_PULLBACK_PCT >= 0.05 else Config.BASKET_TRAILING_PULLBACK_PCT
        f_stop = Config.BASKET_STOP_LOSS_PCT / 100.0 if abs(Config.BASKET_STOP_LOSS_PCT) >= 0.05 else Config.BASKET_STOP_LOSS_PCT
        f_breakeven_trigger = Config.BASKET_BREAKEVEN_TRIGGER_PCT / 100.0 if Config.BASKET_BREAKEVEN_TRIGGER_PCT >= 0.05 else Config.BASKET_BREAKEVEN_TRIGGER_PCT
        f_breakeven_profit = Config.BASKET_BREAKEVEN_PROFIT_PCT / 100.0 if Config.BASKET_BREAKEVEN_PROFIT_PCT >= 0.05 else Config.BASKET_BREAKEVEN_PROFIT_PCT

        alvo_dinamico = total_margin * f_target
        pullback_dinamico = total_margin * f_pullback
        stop_dinamico = total_margin * f_stop # f_stop já traz o sinal negativo correto

        gatilho_breakeven = total_margin * f_breakeven_trigger
        lucro_garantido_breakeven = total_margin * f_breakeven_profit
        passo_avanco = total_margin * 0.0025 # degraus elásticos de 0.25% de margem

        if total_net_pnl > self.max_basket_pnl:
            self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl:
            self.min_basket_pnl = total_net_pnl

        breakeven_ativo = False
        if self.max_basket_pnl >= gatilho_breakeven:
            lucro_excedente = self.max_basket_pnl - gatilho_breakeven
            degraus_avancados = math.floor(lucro_excedente / passo_avanco) if passo_avanco > 0 else 0
            stop_dinamico = lucro_garantido_breakeven + (degraus_avancados * (total_margin * 0.00125)) # garante mais 0.125% de lucro por degrau
            breakeven_ativo = True

        acao = None
        motivo = ""
        lucro_final = total_net_pnl

        if total_net_pnl <= stop_dinamico:
            acao = "FECHAR_TUDO"
            if breakeven_ativo:
                motivo = f"🔵 CATRACA MÓVEL ACIONADA [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\nProteção elástica executada.\nResultado líquido: ${lucro_final:.2f}"
            else:
                motivo = f"🛑 STOP LOSS DO BALDE ACIONADO [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\nCortando perdas líquidas agregadas em: ${lucro_final:.2f}"

        elif self.max_basket_pnl >= alvo_dinamico and (self.max_basket_pnl - total_net_pnl) >= pullback_dinamico:
            acao = "FECHAR_TUDO"
            motivo = f"🎯 TRAILING GLOBAL ATIVADO [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\nEsvaziando balde adaptativo.\nResultado LÍQUIDO líquido: ${lucro_final:.2f}"

        if acao == "FECHAR_TUDO":
            if not Config.OPERA_CONTA_REAL:
                self.simulated_banca += lucro_final

            banca_atual = await self.execution.get_equity() if Config.OPERA_CONTA_REAL else self.simulated_banca
            msg = f"{motivo}\nSaldo da banca: ${banca_atual:.2f}"
            await TelegramLogger.send(msg)
            logging.info(msg.replace("\n", " - "))

            for p in positions:
                symbol = p['symbol']
                side = p['side']
                amount = float(p.get('contracts', 0))
                if amount > 0:
                    logging.info(f"Fechando {symbol} ({side}) - Chase")
                    await self.execution.close_position_limit_chase(symbol, side, amount)

                await self.db.remove_trade(symbol)
                await self.db.add_history(symbol, side, lucro_final / len(positions), 'BASKET_CLOSE')

            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

        elif acao is None and (now - self.last_summary_time >= 600):
            self.last_summary_time = now
            modo_texto = "CONTA REAL" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO"
            status_catraca = "🟢 ATIVA" if breakeven_ativo else "🔴 INATIVA"

            resumo_msg = (
                f"⏱️ <b>RAIO-X DO BALDE (10 min)</b> [{modo_texto}]\n\n"
                f"🔹 <b>Operações:</b> {len(positions)}/{Config.MAX_OPEN_TRADES}\n"
                f"🔹 <b>Margem Alocada:</b> ${total_margin:.2f}\n"
                f"🔹 <b>PnL Atual:</b> ${total_net_pnl:.2f}\n\n"
                f"📈 <b>Topo (Max PnL):</b> ${self.max_basket_pnl:.2f}\n"
                f"📉 <b>Fundo (Min PnL):</b> ${self.min_basket_pnl:.2f}\n\n"
                f"🎯 <b>Alvo Global:</b> ${alvo_dinamico:.2f}\n"
                f"🛑 <b>Stop Global:</b> ${stop_dinamico:.2f}\n"
                f"🔒 <b>Catraca:</b> {status_catraca}"
            )
            await TelegramLogger.send(resumo_msg)

    async def execute_signals(self):
        signals = await self.db.get_elite_signals()
        if not signals: return

        open_trades = await self.db.get_all_open_trades()
        open_symbols = {t[0] for t in open_trades}

        if len(open_symbols) >= Config.MAX_OPEN_TRADES: return

        positions = await self.execution.get_current_positions(self.db)
        now = time.time()

        # 🛡️ TRAVA DE ANTI-CORRELAÇÃO DE MARGEM (ANTI-SUICÍDIO)
        # Conta a direção das posições ativas de forma rigorosa
        buy_positions_count = sum(1 for p in positions if p['side'] in ['LONG', 'BUY'])
        sell_positions_count = sum(1 for p in positions if p['side'] in ['SHORT', 'SELL'])

        # Mapeia a direção do BTC uma única vez no ciclo para otimizar requisições
        btc_trend = await self.check_btc_trend_1h()

        for signal in signals:
            symbol = signal['symbol']

            if symbol in open_symbols: continue

            tempo_liberacao = self.cooldown_memoria.get(symbol, 0)
            if now < tempo_liberacao: continue

            if len(open_symbols) >= Config.MAX_OPEN_TRADES: break

            direction = signal['direction']

            # 🛡️ VALIDAÇÃO DA TRAVA ANTI-SUICÍDIO (MÁXIMO 3 NA MESMA DIREÇÃO NO BALDE)
            if direction == 'BUY' and buy_positions_count >= 3:
                logging.info(f"🚫 [TRAVA ANTI-SUICÍDIO] Compra de {symbol} bloqueada. Limite direcional atingido (já existem {buy_positions_count} posições BUY no balde).")
                continue
            if direction == 'SELL' and sell_positions_count >= 3:
                logging.info(f"🚫 [TRAVA ANTI-SUICÍDIO] Venda de {symbol} bloqueada. Limite direcional atingido (já existem {sell_positions_count} posições SELL no balde).")
                continue

            # FILTRO 1: BÚSSOLA DIRECIONAL DO BITCOIN (1h)
            if direction == 'BUY' and btc_trend == 'BEARISH':
                logging.info(f"🚫 [FILTRO BTC] Compra em {symbol} descartada (BTC em tendência de QUEDA no 1h).")
                continue
            if direction == 'SELL' and btc_trend == 'BULLISH':
                logging.info(f"🚫 [FILTRO BTC] Venda em {symbol} descartada (BTC em tendência de ALTA no 1h).")
                continue

            # FILTRO 2: ALINHAMENTO DE MÚLTIPLOS TIMEFRAMES (4h)
            asset_trend_4h = await self.check_asset_trend_4h(symbol)
            if direction == 'BUY' and asset_trend_4h == 'BEARISH':
                logging.info(f"🚫 [FILTRO 4H] Compra em {symbol} descartada (Timeframe macro 4h é de QUEDA).")
                continue
            if direction == 'SELL' and asset_trend_4h == 'BULLISH':
                logging.info(f"🚫 [FILTRO 4H] Venda em {symbol} descartada (Timeframe macro 4h é de ALTA).")
                continue

            # 🛡️ FILTRO 2B: ALINHAMENTO DE TIMEFRAME 1H ESTRITO (DA PRÓPRIA MOEDA)
            asset_trend_1h = await self.check_asset_trend_1h(symbol)
            if direction == 'BUY' and asset_trend_1h == 'BEARISH':
                logging.info(f"🚫 [FILTRO 1H ATIVO] Compra em {symbol} descartada (Timeframe macro 1h do ativo é de QUEDA).")
                continue
            if direction == 'SELL' and asset_trend_1h == 'BULLISH':
                logging.info(f"🚫 [FILTRO 1H ATIVO] Venda em {symbol} descartada (Timeframe macro 1h do ativo é de ALTA).")
                continue

            # FILTRO 3: VALIDAÇÃO DE LIQUIDEZ POR OPEN INTEREST
            oi_healthy = await self.check_open_interest_healthy(symbol)
            if not oi_healthy:
                logging.info(f"🚫 [FILTRO OI] Ordem em {symbol} abortada. Sem volume de Open Interest institucional ativo.")
                continue

            strategy = self.route_and_calculate_strategy(signal)
            amount_to_invest = strategy["invest_amount"]

            ticker = await asyncio.to_thread(self.public_exchange.fetch_ticker, symbol)
            current_price = float(ticker.get('last') or ticker.get('close') or signal['price'])

            if current_price <= 0: continue

            qty = (amount_to_invest * Config.ALAVANCAGEM) / current_price

            tp_price = current_price * strategy["tp_factor"] if direction == 'BUY' else current_price / strategy["tp_factor"]
            sl_price = current_price * strategy["sl_factor"] if direction == 'BUY' else current_price / strategy["sl_factor"]

            try:
                if Config.OPERA_CONTA_REAL:
                    limit_price = float(ticker['bid']) if direction == 'BUY' else float(ticker['ask'])
                    order = await self.execution.open_position_limit(symbol, direction, qty, limit_price)
                    if not order: continue

                await self.db.add_trade(symbol, direction, current_price, sl_price, tp_price, qty, 1)
                open_symbols.add(symbol)

                # Incrementa o contador para barrar correlações no mesmo ciclo de sinais
                if direction == 'BUY':
                    buy_positions_count += 1
                else:
                    sell_positions_count += 1

                open_trades_after = await self.db.get_all_open_trades()
                total_margin = sum((float(t[3]) * float(t[2])) / Config.ALAVANCAGEM for t in open_trades_after)

                # Alinha os alvos imediatos com a correção percentual
                alvo_atual = total_margin * f_target
                stop_atual = total_margin * f_stop

                banca_atual = await self.execution.get_equity() if Config.OPERA_CONTA_REAL else self.simulated_banca

                modo_texto = "CONTA REAL" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO"
                msg = (
                    f"🔬 [{modo_texto}] ORDEM DETECTADA | {strategy['vertente']}\n\n"
                    f"Ativo: {symbol}, direção: {direction}\n"
                    f"Valor alocado: ${amount_to_invest:.2f} ({strategy['lote_tipo']})\n"
                    f"Score: {signal['score']:.1f}/10.0 | probabilidade: {signal['prob']:.1f}%\n"
                    f"Ml: {signal['reasoning']}\n"
                    f"Saldo da banca: ${banca_atual:.2f}\n\n"
                    f"📊 <b>STATUS DO BALDE:</b>\n"
                    f"Margem Total: ${total_margin:.2f}\n"
                    f"Alvo TP: ${alvo_atual:.2f}\n"
                    f"Risco SL: ${stop_atual:.2f}"
                )

                await TelegramLogger.send(msg)
                logging.info(f"Ordem aberta ({modo_texto}): {symbol} {direction} - Tipo: {strategy['lote_tipo']} - Preço: {current_price}")

                self.cooldown_memoria[symbol] = now + (60 * 60)

            except Exception as e:
                logging.error(f"Erro ao executar sinal {symbol}: {e}")

    async def start_execution_loop(self):
        logging.info("🚀 PREDADOR QUANTITATIVO ONLINE")

        await asyncio.to_thread(self.execution.init_markets_sync)

        modo = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO/DEMO 🔬"
        msg_inicio = f"🚀 PREDADOR QUANTITATIVO ONLINE\nO motor executor foi iniciado com sucesso!\nModo Operacional: {modo}\nAlavancagem Fixa: {Config.ALAVANCAGEM}x\nLimite do Balde: {Config.MAX_OPEN_TRADES} trades simultâneos.\nSaldo Inicial: ${Config.BANCA_DEMO_INICIAL:.2f}"
        await TelegramLogger.send(msg_inicio)

        while True:
            try:
                await self.reconcile_positions()
                await self.cancel_old_pending_orders()
                await self.execute_signals()
                await self.manage_basket()
            except Exception as e:
                logging.error(f"Erro no loop principal do executor: {e}")
            finally:
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        executor = EngineExecutor()
        asyncio.run(executor.start_execution_loop())
    except KeyboardInterrupt:
        logging.info("🛑 Executor encerrado pelo usuário.")
