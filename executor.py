import sys
import os
import asyncio
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

                tickers = await asyncio.to_thread(self.public_exchange.fetch_tickers, [t[0] for t in open_trades])
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

    async def close_position_limit_chase(self, symbol: str, side: str, amount: float, max_retries: int = 3):
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
                await asyncio.sleep(0.5)

                fetched_order = await asyncio.to_thread(self.private_exchange.fetch_order, order['id'], symbol, params={'acknowledged': True})
                if fetched_order.get('status') == 'closed': return True
                else:
                    await asyncio.to_thread(self.private_exchange.cancel_order, order['id'], symbol)
                    await asyncio.sleep(0.2)
                    fetched_order_after = await asyncio.to_thread(self.private_exchange.fetch_order, order['id'], symbol, params={'acknowledged': True})
                    filled = float(fetched_order_after.get('filled', 0.0))
                    remaining_amount = amount - filled
                    if remaining_amount <= 0.00001: return True
            except Exception:
                await asyncio.sleep(0.5)

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
            if sl_price is not None: params['stopLoss'] = str(sl_price)
            if tp_price is not None: params['takeProfit'] = str(tp_price)

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
        self.simulated_banca = float(Config.BANCA_DEMO_INICIAL)

    async def force_close_position(self, symbol: str, side: str, amount: float, outcome: str, pnl: float):
        """Interface direta de zeragem comandada pelo gerenciador."""
        await self.execution.close_position_limit_chase(symbol, side, amount)
        await self.db.remove_trade(symbol)
        if not Config.OPERA_CONTA_REAL: self.simulated_banca += pnl
        await self.db.add_history(symbol, side, pnl, outcome)

    async def order_router_inbound(self, order_packet: dict, signal_packet: dict):
        """Roteador cego: Apenas recebe ordens calculadas e as envia à Bybit."""
        symbol = order_packet['symbol']
        direction = order_packet['direction']
        qty = order_packet['qty']
        current_price = order_packet['current_price']
        tp_price = order_packet['tp']
        sl_price = order_packet['sl']

        try:
            if Config.OPERA_CONTA_REAL:
                ticker = await asyncio.to_thread(self.public_exchange.fetch_ticker, symbol)
                limit_price = float(ticker['bid']) if direction == 'BUY' else float(ticker['ask'])
                order = await self.execution.open_position_limit(symbol, direction, qty, limit_price, sl_price, tp_price)
                if not order: return False

            await self.db.add_trade(symbol, direction, current_price, sl_price, tp_price, qty, 1)
            await self.db.set_cooldown(symbol, time.time() + (60 * 60))
            return True
        except Exception as e:
            logging.error(f"Falha de roteamento crítico em {symbol}: {e}")
            return False
