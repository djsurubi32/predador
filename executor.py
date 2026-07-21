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
import numpy as np
import pandas as pd
from config import Config
from oraculo import OraculoBinance

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

    async def set_leverage(self, symbol: str, leverage: int = 35):
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
            order_side = 'sell' if side.upper() in ['LONG', 'BUY'] else 'buy'
            amount_str = self.private_exchange.amount_to_precision(symbol, amount)
            await asyncio.to_thread(self.private_exchange.create_order, symbol, 'market', order_side, float(amount_str), None, {'reduceOnly': True})
            return True
        except Exception as e:
            logging.error(f"Erro ao fechar posição a mercado {symbol}: {e}")
            return False

    async def close_position_limit_chase(self, symbol: str, side: str, amount: float, max_retries: int = 5):
        if not self.private_exchange or not self.is_real_mode: return True
        order_side = 'sell' if side.upper() in ['LONG', 'BUY'] else 'buy'
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

    async def open_position_limit(self, symbol: str, side: str, amount: float, limit_price: float, sl_price: float = None, tp_price: float = None):
        if not self.private_exchange or not self.is_real_mode: return {"status": "simulated"}
        try:
            await self.set_leverage(symbol, Config.ALAVANCAGEM)
            order_side = 'buy' if side.upper() == 'BUY' else 'sell'
            amount_str = self.private_exchange.amount_to_precision(symbol, amount)
            price_str = self.private_exchange.price_to_precision(symbol, limit_price)
            
            params = {}
            if sl_price is not None:
                params['stopLoss'] = str(sl_price)
            if tp_price is not None:
                params['takeProfit'] = str(tp_price)

            order = await asyncio.to_thread(self.private_exchange.create_order, symbol, 'limit', order_side, float(amount_str), float(price_str), params)
            return order
        except Exception as e:
            logging.error(f"Erro ao abrir posição Limit em {symbol}: {e}")
            raise

class Database:
    def __init__(self, db_name="predador_v31.db"):
        self.db_name = db_name
        self._create_tables()

    def _get_connection(self):
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
            cursor.execute('''CREATE TABLE IF NOT EXISTS cooldowns (
                symbol TEXT PRIMARY KEY, release_time REAL)''')
            conn.commit()

    def _clear_simulation_state_sync(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM trades')
            cursor.execute('DELETE FROM history')
            conn.commit()

    async def clear_simulation_state(self):
        await asyncio.to_thread(self._clear_simulation_state_sync)

    def _set_cooldown_sync(self, symbol, release_time):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO cooldowns (symbol, release_time) VALUES (?, ?)', (symbol, release_time))
            conn.commit()

    async def set_cooldown(self, symbol, release_time):
        await asyncio.to_thread(self._set_cooldown_sync, symbol, release_time)

    def _get_cooldown_sync(self, symbol):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT release_time FROM cooldowns WHERE symbol = ?', (symbol,))
            row = cursor.fetchone()
            return row[0] if row else 0.0

    async def get_cooldown(self, symbol):
        return await asyncio.to_thread(self._get_cooldown_sync, symbol)

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
            cursor.execute('SELECT symbol, side, entry, qty, open_time, sl, tp FROM trades')
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
        
        self.oraculo = OraculoBinance()

        self.max_basket_pnl = 0.0
        self.min_basket_pnl = 0.0
        self.last_empty_heartbeat = time.time()
        self.last_summary_time = time.time()
        self.simulated_banca = float(Config.BANCA_DEMO_INICIAL)

    def route_and_calculate_strategy(self, opp):
        score = float(opp['score'])
        prob = float(opp['prob'])

        reasoning_clean = opp['reasoning'].replace(" ", "")
        is_fluxo_maximo = "Fluxo:3.0" in reasoning_clean
        is_espaco_expandido = "ML_Espaço:3.0" in reasoning_clean

        # 🛡️ AJUSTE ESTRUTURAL (Alavancagem 35x): Distâncias encurtadas para proteger contra liquidação
        if score >= 9.0 and prob >= 75.0:
            return {
                "vertente": "QUALIDADE EXTREMA (SNIPER)",
                "lote_tipo": "Lote Sniper",
                "invest_amount": 6.0,
                "tp_factor": 1.025,  # Busca 2.5% de movimento na moeda (ROE 87.5% a 35x)
                "sl_factor": 0.990   # Tolera apenas 1.0% de oscilação contra (perda de 35% da margem)
            }
        elif is_fluxo_maximo:
            return {
                "vertente": "SCALPING DE MOMENTUM",
                "lote_tipo": "Lote Padrão",
                "invest_amount": 4.0,
                "tp_factor": 1.015,  # Busca 1.5% de movimento (ROE 52.5% a 35x)
                "sl_factor": 0.992   # Tolera apenas 0.8% de oscilação contra
            }
        elif is_espaco_expandido:
            return {
                "vertente": "DAY TRADE DE EXPANSÃO",
                "lote_tipo": "Lote Leve",
                "invest_amount": 3.0,
                "tp_factor": 1.020,  # Busca 2.0% de movimento (ROE 70% a 35x)
                "sl_factor": 0.988   # Tolera 1.2% de oscilação contra
            }

        return {
            "vertente": "PADRÃO ADAPTATIVO",
            "lote_tipo": "Lote de Teste",
            "invest_amount": 2.0,
            "tp_factor": 1.012,  # Busca 1.2% de movimento (ROE 42% a 35x)
            "sl_factor": 0.993   # Tolera 0.7% de oscilação contra
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

    async def analisar_imbalance_l2_bybit(self, symbol: str) -> float:
        try:
            ob = await asyncio.to_thread(self.public_exchange.fetch_order_book, symbol, limit=50)
            bids = np.array(ob['bids'])
            asks = np.array(ob['asks'])

            vol_bids = np.sum(bids[:, 0] * bids[:, 1]) if len(bids) > 0 else 0.0
            vol_asks = np.sum(asks[:, 0] * asks[:, 1]) if len(asks) > 0 else 0.0

            imbalance = (vol_bids - vol_asks) / (vol_bids + vol_asks + 1e-9)
            return float(imbalance)
        except Exception as e:
            logging.error(f"Falha ao ler L2 Bybit para {symbol}: {e}")
            return 0.0

    async def analisar_delta_volume_bybit(self, symbol: str) -> float:
        try:
            ohlcv = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, symbol, '1m', limit=15)
            if not ohlcv or len(ohlcv) < 5:
                return 0.0
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['dir'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
            df['vol_agressao'] = df['volume'] * df['dir']
            
            delta_total = df['vol_agressao'].sum()
            volume_total = df['volume'].sum()

            cvd_ratio = delta_total / (volume_total + 1e-9)
            return float(cvd_ratio)
        except Exception as e:
            logging.error(f"Falha ao ler Fluxo Bybit para {symbol}: {e}")
            return 0.0

    async def validar_microestrutura_bybit(self, symbol: str, direction: str) -> tuple[bool, str]:
        try:
            tarefa_l2 = self.analisar_imbalance_l2_bybit(symbol)
            tarefa_cvd = self.analisar_delta_volume_bybit(symbol)
            
            imbalance_l2, cvd_ratio = await asyncio.gather(tarefa_l2, tarefa_cvd)

            aprovado = False
            motivo = ""

            if direction == 'BUY':
                if imbalance_l2 >= -0.05 and cvd_ratio >= -0.05:
                    aprovado = True
                    motivo = f"L2: {imbalance_l2:.2f} | CVD: {cvd_ratio:.2f}"
                else:
                    motivo = f"Bloqueio Local (Muralha: {imbalance_l2:.2f} / Despejo: {cvd_ratio:.2f})"
            elif direction == 'SELL':
                if imbalance_l2 <= 0.05 and cvd_ratio <= 0.05:
                    aprovado = True
                    motivo = f"L2: {imbalance_l2:.2f} | CVD: {cvd_ratio:.2f}"
                else:
                    motivo = f"Bloqueio Local (Muralha: {imbalance_l2:.2f} / Absorção: {cvd_ratio:.2f})"

            return aprovado, motivo
        except Exception as e:
            logging.error(f"Erro no Batedor Local Bybit para {symbol}: {e}")
            return True, "Bybit Local Indisponível - Bypass Automático"

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

    async def monitor_individual_positions(self):
        try:
            positions = await self.execution.get_current_positions(self.db)
            if not positions:
                return

            open_trades = await self.db.get_all_open_trades()
            trade_dict = {t[0]: t for t in open_trades}

            for p in positions:
                symbol = p['symbol']
                if symbol not in trade_dict:
                    continue

                trade = trade_dict[symbol]
                side = trade[1].upper()
                sl_price = float(trade[5])
                tp_price = float(trade[6])
                
                ticker = await asyncio.to_thread(self.public_exchange.fetch_ticker, symbol)
                current_price = float(ticker.get('last') or ticker.get('close'))

                fechar = False
                motivo = ""

                if side in ['LONG', 'BUY']:
                    if current_price >= tp_price:
                        fechar = True
                        motivo = "🎯 TAKE PROFIT INDIVIDUAL ATINGIDO"
                    elif current_price <= sl_price:
                        fechar = True
                        motivo = "🛑 STOP LOSS INDIVIDUAL ATINGIDO"
                else:
                    if current_price <= tp_price:
                        fechar = True
                        motivo = "🎯 TAKE PROFIT INDIVIDUAL ATINGIDO"
                    elif current_price >= sl_price:
                        fechar = True
                        motivo = "🛑 STOP LOSS INDIVIDUAL ATINGIDO"

                if fechar:
                    amount = float(p.get('contracts', 0))
                    if amount > 0:
                        logging.info(f"Fechando moeda isolada {symbol} ({side}) - {motivo}")
                        await self.execution.close_position_market(symbol, side, amount)
                    
                    pnl_realizado = float(p.get('netPnl', 0))
                    await self.db.remove_trade(symbol)
                    
                    if not Config.OPERA_CONTA_REAL:
                        self.simulated_banca += pnl_realizado
                    
                    await self.db.add_history(symbol, side, pnl_realizado, motivo)

                    banca_atual = await self.execution.get_equity() if Config.OPERA_CONTA_REAL else self.simulated_banca
                    msg = (
                        f"{motivo} [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\n"
                        f"Ativo: {symbol}\n"
                        f"Direção: {side}\n"
                        f"PnL Líquido: ${pnl_realizado:.2f}\n"
                        f"Saldo Atual: ${banca_atual:.2f}"
                    )
                    await TelegramLogger.send(msg)
                    logging.info(msg.replace("\n", " - "))
        except Exception as e:
            logging.error(f"Erro no monitoramento individual de posições: {e}")

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

        if total_net_pnl > self.max_basket_pnl:
            self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl:
            self.min_basket_pnl = total_net_pnl

        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        # 🛡️ VERTENTE A: TRAILING STOP INSTITUCIONAL EM FASES (Para 35x)
        # Substitui a catraca covarde por estrangulamento agressivo de ROE (Return On Equity)
        
        stop_dinamico_usd = -total_margin * 0.40  # Hard stop da cesta (Limite máximo de 40% de perda sobre a margem)
        fase_catraca = "INATIVA"
        
        if max_roe >= 0.50:
            # FASE 3 (Asfixia Extrema): Acima de 50% de ROE, permite apenas 5% de recuo absoluto.
            stop_dinamico_usd = (max_roe - 0.05) * total_margin
            fase_catraca = "ASFIXIA (Fase 3)"
        elif max_roe >= 0.30:
            # FASE 2 (Fixação): Acima de 30% de ROE, permite 10% de recuo.
            stop_dinamico_usd = (max_roe - 0.10) * total_margin
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.15:
            # FASE 1 (Break-Even Dinâmico): Acima de 15% de ROE, trava em lucro garantido (+5% ROE).
            stop_dinamico_usd = 0.05 * total_margin
            fase_catraca = "BREAK-EVEN (Fase 1)"

        acao = None
        motivo = ""
        lucro_final = total_net_pnl

        if total_net_pnl <= stop_dinamico_usd:
            acao = "FECHAR_TUDO"
            if fase_catraca != "INATIVA":
                motivo = (
                    f"🔵 CATRACA MÓVEL ACIONADA [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\n"
                    f"Fase Ativa: {fase_catraca}\n"
                    f"Proteção elástica executada com precisão no ROE.\n"
                    f"Resultado líquido: ${lucro_final:.2f}"
                )
            else:
                motivo = (
                    f"🛑 STOP LOSS DO BALDE ACIONADO [{ 'CONTA REAL' if Config.OPERA_CONTA_REAL else 'SIMULAÇÃO' }]\n"
                    f"Cortando perdas agregadas (Hard Stop de Proteção).\n"
                    f"Resultado líquido: ${lucro_final:.2f}"
                )

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
                    logging.info(f"Fechando {symbol} ({side}) - Catraca Global")
                    await self.execution.close_position_limit_chase(symbol, side, amount)

                await self.db.remove_trade(symbol)
                await self.db.add_history(symbol, side, lucro_final / len(positions), 'BASKET_CLOSE')

            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

        elif acao is None and (now - self.last_summary_time >= 600):
            self.last_summary_time = now
            modo_texto = "CONTA REAL" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO"
            
            resumo_msg = (
                f"⏱️ <b>RAIO-X DO BALDE (10 min)</b> [{modo_texto}]\n\n"
                f"🔹 <b>Operações:</b> {len(positions)}/{Config.MAX_OPEN_TRADES}\n"
                f"🔹 <b>Margem Alocada:</b> ${total_margin:.2f}\n"
                f"🔹 <b>PnL Atual:</b> ${total_net_pnl:.2f} ({current_roe * 100:.1f}% ROE)\n\n"
                f"📈 <b>Topo (Max PnL):</b> ${self.max_basket_pnl:.2f} ({max_roe * 100:.1f}% ROE)\n"
                f"📉 <b>Fundo (Min PnL):</b> ${self.min_basket_pnl:.2f}\n\n"
                f"🔒 <b>Catraca:</b> {fase_catraca}\n"
                f"🛑 <b>Gatilho de Fechamento em:</b> ${stop_dinamico_usd:.2f}"
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

        buy_positions_count = sum(1 for p in positions if p['side'] in ['LONG', 'BUY'])
        sell_positions_count = sum(1 for p in positions if p['side'] in ['SHORT', 'SELL'])

        btc_trend = await self.check_btc_trend_1h()

        for signal in signals:
            symbol = signal['symbol']

            if symbol in open_symbols: continue

            tempo_liberacao = await self.db.get_cooldown(symbol)
            if now < tempo_liberacao: continue

            if len(open_symbols) >= Config.MAX_OPEN_TRADES: break

            direction = signal['direction']

            if direction == 'BUY' and buy_positions_count >= 3:
                logging.info(f"🚫 [TRAVA ANTI-SUICÍDIO] Compra de {symbol} bloqueada. Limite direcional atingido.")
                continue
            if direction == 'SELL' and sell_positions_count >= 3:
                logging.info(f"🚫 [TRAVA ANTI-SUICÍDIO] Venda de {symbol} bloqueada. Limite direcional atingido.")
                continue

            if direction == 'BUY' and btc_trend == 'BEARISH':
                logging.info(f"🚫 [FILTRO BTC] Compra em {symbol} descartada (BTC em QUEDA no 1h).")
                continue
            if direction == 'SELL' and btc_trend == 'BULLISH':
                logging.info(f"🚫 [FILTRO BTC] Venda em {symbol} descartada (BTC em ALTA no 1h).")
                continue

            asset_trend_4h = await self.check_asset_trend_4h(symbol)
            if direction == 'BUY' and asset_trend_4h == 'BEARISH':
                logging.info(f"🚫 [FILTRO 4H] Compra em {symbol} descartada (Macro 4h é de QUEDA).")
                continue
            if direction == 'SELL' and asset_trend_4h == 'BULLISH':
                logging.info(f"🚫 [FILTRO 4H] Venda em {symbol} descartada (Macro 4h é de ALTA).")
                continue

            asset_trend_1h = await self.check_asset_trend_1h(symbol)
            if direction == 'BUY' and asset_trend_1h == 'BEARISH':
                logging.info(f"🚫 [FILTRO 1H] Compra em {symbol} descartada (Macro 1h é de QUEDA).")
                continue
            if direction == 'SELL' and asset_trend_1h == 'BULLISH':
                logging.info(f"🚫 [FILTRO 1H] Venda em {symbol} descartada (Macro 1h é de ALTA).")
                continue

            oi_healthy = await self.check_open_interest_healthy(symbol)
            if not oi_healthy:
                logging.info(f"🚫 [FILTRO OI] Ordem em {symbol} abortada. Sem volume Open Interest.")
                continue

            oraculo_aprovado, oraculo_motivo = await self.oraculo.validar_sinal_institucional(symbol, direction)
            if not oraculo_aprovado:
                logging.info(f"🚫 [ORÁCULO BINANCE] Ordem em {symbol} ({direction}) bloqueada pela Binance. Motivo: {oraculo_motivo}")
                continue

            bybit_aprovado, bybit_motivo = await self.validar_microestrutura_bybit(symbol, direction)
            if not bybit_aprovado:
                logging.info(f"🚫 [BATEDOR BYBIT] Ordem em {symbol} ({direction}) bloqueada localmente. Motivo: {bybit_motivo}")
                continue

            strategy = self.route_and_calculate_strategy(signal)
            amount_to_invest = strategy["invest_amount"]

            ticker = await asyncio.to_thread(self.public_exchange.fetch_ticker, symbol)
            current_price = float(ticker.get('last') or ticker.get('close') or signal['price'])

            if current_price <= 0: continue

            qty = (amount_to_invest * Config.ALAVANCAGEM) / current_price

            # 🛡️ REVERSÃO MATEMÁTICA CORRETA PARA SHORTS (Bidirecionalidade Cripto)
            if direction == 'BUY':
                tp_price = current_price * strategy["tp_factor"]
                sl_price = current_price * strategy["sl_factor"]
            else:
                dist_tp = strategy["tp_factor"] - 1.0
                dist_sl = 1.0 - strategy["sl_factor"]
                tp_price = current_price * (1.0 - dist_tp)
                sl_price = current_price * (1.0 + dist_sl)

            try:
                if Config.OPERA_CONTA_REAL:
                    limit_price = float(ticker['bid']) if direction == 'BUY' else float(ticker['ask'])
                    order = await self.execution.open_position_limit(symbol, direction, qty, limit_price, sl_price, tp_price)
                    if not order: continue

                await self.db.add_trade(symbol, direction, current_price, sl_price, tp_price, qty, 1)
                open_symbols.add(symbol)
                
                if direction == 'BUY':
                    buy_positions_count += 1
                else:
                    sell_positions_count += 1

                banca_atual = await self.execution.get_equity() if Config.OPERA_CONTA_REAL else self.simulated_banca

                modo_texto = "CONTA REAL" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO"
                msg = (
                    f"🔬 [{modo_texto}] ORDEM DETECTADA | {strategy['vertente']}\n\n"
                    f"Ativo: {symbol} | Direção: {direction}\n"
                    f"Valor alocado: ${amount_to_invest:.2f} ({strategy['lote_tipo']})\n"
                    f"Score: {signal['score']:.1f}/10.0 | Probabilidade: {signal['prob']:.1f}%\n"
                    f"Ml: {signal['reasoning']}\n"
                    f"Oráculo Binance: {oraculo_motivo}\n"
                    f"Batedor Bybit: Validado Local ({bybit_motivo})\n"
                    f"Saldo da banca: ${banca_atual:.2f}"
                )

                await TelegramLogger.send(msg)
                logging.info(f"Ordem aberta ({modo_texto}): {symbol} {direction} - Tipo: {strategy['lote_tipo']} - Preço: {current_price}")

                await self.db.set_cooldown(symbol, now + (60 * 60))

            except Exception as e:
                logging.error(f"Erro ao executar sinal {symbol}: {e}")

    async def start_execution_loop(self):
        logging.info("🚀 PREDADOR QUANTITATIVO ONLINE")

        if not Config.OPERA_CONTA_REAL:
            logging.info("🧹 Modo SIMULAÇÃO detectado. Executando limpeza de operações fantasmas e histórico do banco de dados...")
            await self.db.clear_simulation_state()

        await asyncio.to_thread(self.execution.init_markets_sync)

        modo = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO/DEMO 🔬"
        msg_inicio = (
            f"🚀 PREDADOR QUANTITATIVO ONLINE\n"
            f"O motor executor bidirecional foi iniciado com sucesso!\n"
            f"Modo Operacional: {modo}\n"
            f"Alavancagem Fixa: {Config.ALAVANCAGEM}x\n"
            f"Limite do Balde: {Config.MAX_OPEN_TRADES} trades simultâneos.\n"
            f"Saldo Inicial: ${Config.BANCA_DEMO_INICIAL:.2f}"
        )
        await TelegramLogger.send(msg_inicio)

        while True:
            try:
                await self.reconcile_positions()
                await self.cancel_old_pending_orders()
                await self.execute_signals()
                await self.monitor_individual_positions()
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
