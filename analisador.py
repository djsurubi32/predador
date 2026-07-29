import sys
import os
import asyncio
import math
import logging
import time
import sqlite3
import joblib
import warnings
import hashlib
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

import aiosqlite
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from numpy.linalg import norm
import feedparser
from sentence_transformers import SentenceTransformer

# Importamos o Config e o novo universe_provider assíncrono
from config import Config, universe_provider
from ta_indicators import add_custom_ta

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass(frozen=True)
class LiquiditySnapshot:
    vwap_imb: float | None
    cvd_imb: float | None
    spoof_buy: bool
    spoof_sell: bool
    stale: bool = False

class Database:
    def __init__(self, db_name="predador_v31.db"):
        self.db_name = getattr(Config, 'DB_NAME', db_name)
        self._create_tables_sync()

    def _create_tables_sync(self):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, entry REAL,
                sl REAL, tp REAL, qty REAL, force INTEGER, open_time REAL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS elite_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, 
                price REAL, prob REAL, score REAL, atr REAL, sma_atr REAL, 
                funding REAL, reasoning TEXT, timestamp REAL)''')
            conn.commit()

    async def update_elite_signals(self, signals):
        now = time.time()
        data_to_insert = [
            (sig['symbol'], sig['direction'], sig['price'], sig['prob'],
             sig['score'], sig['current_atr'], sig['sma_atr'], sig['funding'],
             sig['reasoning'], now) for sig in signals
        ]

        async with aiosqlite.connect(self.db_name, timeout=30) as db:
            await db.execute('PRAGMA journal_mode=WAL;')
            await db.execute('DELETE FROM elite_signals WHERE timestamp < ?', (now - 604800,))
            
            if data_to_insert:
                await db.executemany('''INSERT INTO elite_signals
                    (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data_to_insert)
            await db.commit()

    async def get_open_trades_count(self):
        async with aiosqlite.connect(self.db_name, timeout=30) as db:
            async with db.execute('SELECT COUNT(*) FROM trades') as cursor:
                res = await cursor.fetchone()
                return res[0] if res else 0

class LiquidityCore:
    def __init__(self):
        self.exchanges = {
            "Binance": ccxt_async.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}}),
            "Bybit": ccxt_async.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}}),
        }
        self.ob_history = {name: {} for name in self.exchanges.keys()}

    async def close_exchanges(self):
        for ex in self.exchanges.values():
            await ex.close()

    async def fetch_liquidity_data(self, exchange, ex_name, symbol) -> LiquiditySnapshot:
        try:
            now = time.time()
            ob = await exchange.fetch_order_book(symbol, limit=20)
            vwap_imb, bid_vol, ask_vol = 50.0, 0.0, 0.0

            if ob.get('bids') and ob.get('asks'):
                mid_price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2.0
                bid_vol = sum(vol * math.exp(-50 * abs(price - mid_price) / mid_price) for price, vol in ob['bids'][:10])
                ask_vol = sum(vol * math.exp(-50 * abs(price - mid_price) / mid_price) for price, vol in ob['asks'][:10])
                total = bid_vol + ask_vol
                if total > 0: 
                    vwap_imb = (bid_vol / total) * 100

            spoof_buy, spoof_sell = False, False
            prev = self.ob_history[ex_name].get(symbol)
            
            if prev and (now - prev['timestamp'] < 15.0):
                if prev['bids'] > 0 and (prev['bids'] - bid_vol) / prev['bids'] > 0.35: 
                    spoof_buy = True
                if prev['asks'] > 0 and (prev['asks'] - ask_vol) / prev['asks'] > 0.35: 
                    spoof_sell = True

            self.ob_history[ex_name][symbol] = {'bids': bid_vol, 'asks': ask_vol, 'timestamp': now}

            cvd_imb = 50.0
            try:
                trades = await exchange.fetch_trades(symbol, limit=200)
                if trades:
                    cvd_buy = sum(t.get('amount', 0) for t in trades if t.get('side') == 'buy')
                    cvd_sell = sum(t.get('amount', 0) for t in trades if t.get('side') == 'sell')
                    if (cvd_buy + cvd_sell) > 0: 
                        cvd_imb = (cvd_buy / (cvd_buy + cvd_sell)) * 100
            except Exception as e:
                cvd_imb = None

            return LiquiditySnapshot(vwap_imb=vwap_imb, cvd_imb=cvd_imb, spoof_buy=spoof_buy, spoof_sell=spoof_sell, stale=False)
        except Exception as e:
            return LiquiditySnapshot(vwap_imb=None, cvd_imb=None, spoof_buy=False, spoof_sell=False, stale=True)

    async def get_liquidity_report(self, symbol) -> dict[str, LiquiditySnapshot]:
        tasks = [self.fetch_liquidity_data(ex, name, symbol) for name, ex in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots = {}
        for name, r in zip(self.exchanges.keys(), results):
            if isinstance(r, Exception):
                snapshots[name] = LiquiditySnapshot(None, None, False, False, True)
            else:
                snapshots[name] = r
        return snapshots

class LocalNewsCore:
    def __init__(self):
        try:
            model_name = getattr(Config, 'NLP_MODEL_NAME', 'ProsusAI/finbert')
            self.model = SentenceTransformer(model_name)
            self.bull_emb = self.model.encode(["bull market positive good news surge adoption breakout"])
            self.bear_emb = self.model.encode(["bear market crash negative bad news regulation hack"])
        except Exception:
            self.model = None
        self.last_fetch = 0
        self.current_sentiment = 0.0

    def fetch_and_score_sync(self):
        if not self.model: 
            return 0.0
        titles = set()
        for url in ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]:
                    titles.add(entry.title)
            except Exception:
                pass
        
        if not titles: 
            return 0.0
            
        try:
            embs = self.model.encode(list(titles))
            bull = np.dot(embs, self.bull_emb.T) / (norm(embs, axis=1, keepdims=True) * norm(self.bull_emb))
            bear = np.dot(embs, self.bear_emb.T) / (norm(embs, axis=1, keepdims=True) * norm(self.bear_emb))
            sentiment = float(np.mean(bull - bear))
            return sentiment
        except Exception:
            return 0.0

    async def get_sentiment_score(self):
        now = time.time()
        if now - self.last_fetch > 300:
            self.current_sentiment = await asyncio.to_thread(self.fetch_and_score_sync)
            self.last_fetch = now
        return self.current_sentiment

class RadarCore:
    def __init__(self):
        self.public_exchange = ccxt_async.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        self.db = Database()
        self.liquidity = LiquidityCore()
        self.news = LocalNewsCore()
        # Inicializa vazio, será populado dinamicamente no loop
        self.next_trade_time = {}
        self.loaded_models = OrderedDict()
        self.MAX_MODELS_IN_RAM = getattr(Config, 'MAX_MODELS_IN_RAM', 30)

    async def shutdown(self):
        await self.public_exchange.close()
        await self.liquidity.close_exchanges()

    def load_brain(self, symbol):
        safe_name = symbol.replace('/', '_').replace(':', '_')
        models_dir = getattr(Config, 'MODELS_DIR', 'modelos_ia')
        path = os.path.join(models_dir, f"{safe_name}.pkl")
        
        if not os.path.exists(path): 
            return None

        file_mod_time = os.path.getmtime(path)
        
        if symbol in self.loaded_models:
            if self.loaded_models[symbol]['mod_time'] >= file_mod_time:
                self.loaded_models.move_to_end(symbol)
                return self.loaded_models[symbol]['brain']
            
        try:
            if len(self.loaded_models) >= self.MAX_MODELS_IN_RAM:
                ejected_symbol, _ = self.loaded_models.popitem(last=False)
            
            brain = joblib.load(path)
            self.loaded_models[symbol] = {'brain': brain, 'mod_time': file_mod_time}
            return brain
        except Exception:
            return None

    def prepare_features(self, df):
        df = add_custom_ta(df)

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema12 - ema26

        df['SMA_ATR_100'] = df.get('ATRr_14', df['close'].rolling(14).std()).rolling(window=100).mean()

        numeric_cols = df.columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        ema5 = df['close'].ewm(span=5, adjust=False).mean()
        df['close_smooth'] = ema5.ewm(span=5, adjust=False).mean()

        df['noise_index'] = abs(df['close'] - df['close_smooth']) / np.where(df['close'] == 0, 1e-5, df['close'])
        df['log_return'] = np.log(df['close'] / np.where(df['close'].shift(1) == 0, 1e-5, df['close'].shift(1)))
        df['volatility_cluster'] = df['log_return'].rolling(window=20).std()

        vol_roll = df['volume'].rolling(20)
        df['vol_zscore'] = (df['volume'] - vol_roll.mean()) / np.where(vol_roll.std() == 0, 1e-5, vol_roll.std())

        df['EMA_20'] = df.get('EMA_20', df['close'].ewm(span=20, adjust=False).mean())
        df['price_vs_ema'] = df['close'] - df['EMA_20']

        bbl = df.get('BBL_20_2.0', df['close'])
        bbu = df.get('BBU_20_2.0', df['close'])
        df['bb_pos'] = (df['close'] - bbl) / np.where((bbu - bbl) == 0, 1e-5, bbu - bbl)

        df['rsi_slope'] = df.get('RSI_14', pd.Series(0, index=df.index)).diff(3)
        df['price_slope'] = df['close_smooth'].diff(3).fillna(0.0)
        df['rsi_divergence'] = np.where((df['price_slope'] < 0) & (df['rsi_slope'] > 0), 1,
                               np.where((df['price_slope'] > 0) & (df['rsi_slope'] < 0), -1, 0))

        df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
        df['cvd'] = (df['volume'] * df['candle_dir']).cumsum()
        df['cvd_trend'] = df['cvd'] - df['cvd'].rolling(20).mean()

        if 'open_interest' in df.columns:
            df['oi_temp'] = df['open_interest'].replace(0, np.nan).ffill().bfill()
            df['oi_change'] = df['oi_temp'].pct_change(fill_method=None).fillna(0.0)
            df['oi_trend'] = df['oi_change'].rolling(window=5).mean().fillna(0.0)
            price_pct = df['close'].pct_change(fill_method=None).fillna(0.0)

            df['oi_price_divergence'] = np.where((price_pct > 0) & (df['oi_change'] > 0), 1.0,
                                        np.where((price_pct < 0) & (df['oi_change'] > 0), -1.0,
                                        np.where((price_pct > 0) & (df['oi_change'] < 0), -0.5,
                                        np.where((price_pct < 0) & (df['oi_change'] < 0), 0.5, 0.0))))
            df.drop(columns=['oi_temp'], inplace=True)
            df['oi_momentum'] = df['open_interest'].diff(5) / np.where(df['open_interest'].rolling(20).mean() == 0, 1e-5, df['open_interest'].rolling(20).mean())
        else:
            df['oi_change'], df['oi_trend'], df['oi_price_divergence'], df['oi_momentum'] = 0.0, 0.0, 0.0, 0.0

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_slope'] = df['obv'].diff(3).fillna(0.0)
        df['obv_rsi_div'] = np.where((df['obv_slope'] > 0) & (df['rsi_slope'] < 0), 1.0,
                            np.where((df['obv_slope'] < 0) & (df['rsi_slope'] > 0), -1.0, 0.0))

        df['cvd_accel'] = df['cvd'].diff(3).diff(3).fillna(0.0)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        df['liq_vacuum'] = ((df['high'] - df['low']) / np.where(df['volume'] == 0, 1e-5, df['volume'])).rolling(10).mean().fillna(0.0)

        if 'funding_rate' in df.columns:
            df['funding_rate'] = df['funding_rate'].replace(0, np.nan).ffill()
            df['funding_rate_delta'] = df['funding_rate'].diff(3)
        else:
            df['funding_rate_delta'] = 0.0

        vol_drop = df['volume'] < df['volume'].shift(1)
        roc = df['close'].pct_change()
        raw_nvi = (vol_drop * roc).cumsum()
        df['nvi'] = raw_nvi - raw_nvi.rolling(100).mean()

        lag1_var = df['log_return'].rolling(20).var()
        lag5_var = df['close'].pct_change(5).rolling(20).var()
        df['hurst_proxy'] = (np.log(np.where(lag5_var == 0, 1e-5, lag5_var)) / 
                             np.log(np.where(lag1_var == 0, 1e-5, lag1_var))).clip(-3.0, 3.0)

        roll50 = df['close'].rolling(50)
        df['z_score_50'] = (df['close'] - roll50.mean()) / np.where(roll50.std() == 0, 1e-5, roll50.std())

        r = df['log_return']
        df['autocorr_3'] = r.rolling(20).corr(r.shift(3))

        n_period = 20
        high_max = df['high'].rolling(n_period).max()
        low_min = df['low'].rolling(n_period).min()
        path_length = np.abs(df['close'].diff()).rolling(n_period).sum()
        df['fractal_dim'] = (np.log(np.where(path_length == 0, 1e-5, path_length)) / 
                             np.log(np.where((high_max - low_min) == 0, 1e-5, (high_max - low_min)))).clip(-3.0, 3.0)

        atr_14 = df.get('ATRr_14', df['close'].rolling(14).std())
        kc_upper = df['EMA_20'] + (1.5 * atr_14)
        kc_lower = df['EMA_20'] - (1.5 * atr_14)
        df['squeeze_ratio'] = (bbu - bbl) / np.where((kc_upper - kc_lower) == 0, 1e-5, kc_upper - kc_lower)

        df['natr'] = (atr_14 / df['close']) * 100
        df['skewness_20'] = df['log_return'].rolling(20).skew()

        ema9_macd = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['macd_hist_vel'] = (df['MACD_12_26_9'] - ema9_macd).diff(2)

        cols_to_drop = ['obv', 'obv_slope']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        df_indexed = df.set_index('datetime')
        
        df_1h = df_indexed['close'].resample('1h', label='right', closed='right').last()
        h1_ema = df_1h.ewm(span=20, adjust=False).mean().shift(1)
        df_indexed['ema_20_1h'] = h1_ema.reindex(df_indexed.index, method='ffill')

        df_4h = df_indexed['close'].resample('4h', label='right', closed='right').last()
        h4_ema = df_4h.ewm(span=20, adjust=False).mean().shift(1)
        df_indexed['ema_20_4h'] = h4_ema.reindex(df_indexed.index, method='ffill')

        df_indexed.reset_index(drop=True, inplace=True)
        df = df_indexed

        df['ema_20_1h'] = df['ema_20_1h'].fillna(df['close'])
        df['ema_20_4h'] = df['ema_20_4h'].fillna(df['close'])
        df['mtf_dist_1h'] = (df['close'] - df['ema_20_1h']) / df['ema_20_1h']
        df['mtf_dist_4h'] = (df['close'] - df['ema_20_4h']) / df['ema_20_4h']

        df.dropna(inplace=True)
        return df

    def predict_with_brain(self, brain, current_data_row, historical_df) -> tuple[Optional[str], float, float]:
        expected_features = brain.get('feature_names', [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14',
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change',
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'noise_index',
            'obv_rsi_div', 'cvd_accel', 'vwap_dist', 'liq_vacuum', 'funding_rate_delta',
            'oi_momentum', 'nvi', 'hurst_proxy', 'z_score_50', 'autocorr_3', 'fractal_dim',
            'squeeze_ratio', 'natr', 'skewness_20', 'macd_hist_vel'
        ])

        missing = [f for f in expected_features if f not in current_data_row or pd.isna(current_data_row[f]) and f != 'hmm_regime']
        if missing:
            return None, 0.0, 0.0

        try:
            X_pred = pd.DataFrame([{f: current_data_row.get(f, 0.0) for f in expected_features}])
            
            if hasattr(brain.get('hmm'), 'predict'):
                try: 
                    seq = historical_df[['log_return','volatility_cluster']].tail(200).values
                    X_pred['hmm_regime'] = int(brain['hmm'].predict(seq)[-1])
                except Exception: 
                    X_pred['hmm_regime'] = 0
            else:
                X_pred['hmm_regime'] = 0

            X_pred = X_pred[expected_features]

            X_meta = np.column_stack((
                brain['lgbm'].predict_proba(X_pred),
                brain['xgb'].predict_proba(X_pred),
                brain['catboost'].predict_proba(X_pred)
            ))
            
            classes = list(brain['meta'].classes_)
            p_dict = dict(zip(classes, brain['meta'].predict_proba(X_meta)[0]))

            prob_alta = p_dict.get(1, 0.0) * 100
            prob_queda = p_dict.get(2, 0.0) * 100

            min_prob_predict = getattr(Config, 'MIN_PROB_PREDICT', 52.0)

            if prob_alta >= min_prob_predict: 
                return "ALTA", prob_alta, prob_queda
            elif prob_queda >= min_prob_predict: 
                return "QUEDA", prob_alta, prob_queda
            else: 
                return "HOLD", prob_alta, prob_queda
        except Exception:
            return None, 0.0, 0.0

    async def analyze_symbol(self, symbol, news_sentiment):
        if time.time() < self.next_trade_time.get(symbol, 0): 
            return None
            
        brain = await asyncio.to_thread(self.load_brain, symbol)
        if not brain: 
            return None

        try:
            timeframe = getattr(Config, 'TIMEFRAME', '5m')
            ohlcv = await self.public_exchange.fetch_ohlcv(symbol, timeframe, limit=400)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df = await asyncio.to_thread(self.prepare_features, df)
            if len(df) < 30: 
                return None

            raw_price = df.iloc[-1]['close']
            current_atr = float(df.iloc[-1].get('ATRr_14', raw_price * 0.005))
            sma_atr = float(df.iloc[-1].get('SMA_ATR_100', current_atr))
            expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0

            bb_pos = float(df.iloc[-1].get('bb_pos', 0.5))

            ml_text, prob_alta, prob_queda = await asyncio.to_thread(
                self.predict_with_brain, brain, df.iloc[-1].to_dict(), df
            )
            
            if ml_text is None:
                return None

            direction = 'BUY' if ml_text == "ALTA" else 'SELL' if ml_text == "QUEDA" else 'HOLD'
            ml_prob = max(prob_alta, prob_queda)
            
            if direction == 'HOLD': 
                return None

            liquidity_snapshots = await self.liquidity.get_liquidity_report(symbol)

            score, force, reasoning = self.calculate_conviction_score(
                direction, ml_prob, liquidity_snapshots, news_sentiment, current_atr, raw_price, bb_pos, expected_move_pct
            )

            if score == 0:
                return None

            return {
                'symbol': symbol, 'direction': direction, 'price': float(raw_price),
                'prob': ml_prob, 'score': score, 'current_atr': current_atr,
                'sma_atr': sma_atr, 'funding': 0.0, 'reasoning': reasoning,
                'expected_move': expected_move_pct
            }
        except Exception:
            return None

    def calculate_conviction_score(self, direction, ml_prob, snapshots: dict[str, LiquiditySnapshot], nlp_sentiment, current_atr, raw_price, bb_pos, expected_move_pct):
        min_liquidity_sources = getattr(Config, 'MIN_LIQUIDITY_SOURCES', 1)
        valid_sources = [s for s in snapshots.values() if not s.stale and s.vwap_imb is not None]
        
        if len(valid_sources) < min_liquidity_sources:
             return 0, 1, f"VETO LIQUIDEZ: Faltam dados válidos ({len(valid_sources)}/{min_liquidity_sources})."

        min_edge = getattr(Config, 'MIN_EDGE_PCT', 0.15)
        if expected_move_pct < min_edge:
             return 0, 1, f"VETO CUSTO: Movimento ({expected_move_pct:.2f}%) menor que limite mínimo ({min_edge}%)."

        bb_top = getattr(Config, 'BB_POS_TOP_VETO', 0.95)
        bb_bot = getattr(Config, 'BB_POS_BOTTOM_VETO', 0.05)
        
        if direction == 'BUY' and bb_pos >= bb_top: return 0, 1, "VETO ESPAÇO RIGIDO: Compra de Topo."
        if direction == 'SELL' and bb_pos <= bb_bot: return 0, 1, "VETO ESPAÇO RIGIDO: Venda de Fundo."

        ml_pts = 0.0
        min_conviction = getattr(Config, 'MIN_PROB_CONVICTION', 52.0)
        
        if ml_prob >= getattr(Config, 'PROB_TIER_4', 90.0): ml_pts = 4.0
        elif ml_prob >= getattr(Config, 'PROB_TIER_3', 85.0): ml_pts = 3.0
        elif ml_prob >= getattr(Config, 'PROB_TIER_2', 75.0): ml_pts = 2.0
        elif ml_prob >= min_conviction: ml_pts = 1.0
        else: return 0, 1, f"VETO ML: Probabilidade Baixa ({ml_prob:.1f}%)."

        espaco_pts = 0.0
        min_move = getattr(Config, 'MIN_EXPECTED_MOVE_PCT', 0.4)
        
        if expected_move_pct >= getattr(Config, 'ALVO_TIER_3', 1.5): espaco_pts = 4.0
        elif expected_move_pct >= getattr(Config, 'ALVO_TIER_2', 0.8): espaco_pts = 3.0
        elif expected_move_pct >= min_move: espaco_pts = 2.0
        else: return 0, 1, "VETO ESPAÇO: Alvo muito curto."

        liq_pts = 0.0
        apoios = []
        
        binance_snap = snapshots.get('Binance')
        bybit_snap = snapshots.get('Bybit')

        if binance_snap and binance_snap.vwap_imb is not None:
            if direction == 'BUY' and binance_snap.vwap_imb >= 50.5: liq_pts += 1.0; apoios.append('Binance_VWAP')
            elif direction == 'SELL' and binance_snap.vwap_imb <= 49.8: liq_pts += 1.0; apoios.append('Binance_VWAP')

        if bybit_snap and bybit_snap.vwap_imb is not None:
            if direction == 'BUY' and bybit_snap.vwap_imb >= 50.5: liq_pts += 1.0; apoios.append('Bybit_VWAP')
            elif direction == 'SELL' and bybit_snap.vwap_imb <= 49.8: liq_pts += 1.0; apoios.append('Bybit_VWAP')

        cvd_buy, cvd_sell = False, False
        if binance_snap and binance_snap.cvd_imb is not None:
            if binance_snap.cvd_imb >= 50.2: cvd_buy = True
            if binance_snap.cvd_imb <= 49.8: cvd_sell = True
        if bybit_snap and bybit_snap.cvd_imb is not None:
            if bybit_snap.cvd_imb >= 50.5: cvd_buy = True
            if bybit_snap.cvd_imb <= 49.8: cvd_sell = True

        if direction == 'BUY' and cvd_buy: liq_pts += 1.0; apoios.append('CVD')
        if direction == 'SELL' and cvd_sell: liq_pts += 1.0; apoios.append('CVD')

        if liq_pts < 1.0: return 0, 1, "VETO VWAP: Sem apoio institucional."

        spoof_buy = any(s.spoof_buy for s in valid_sources)
        spoof_sell = any(s.spoof_sell for s in valid_sources)
        if direction == 'BUY' and spoof_buy: return 0, 1, "VETO SPOOFING."
        if direction == 'SELL' and spoof_sell: return 0, 1, "VETO SPOOFING."

        total = ml_pts + espaco_pts + liq_pts
        detalhes = ", ".join(apoios) if apoios else "Nenhum"
        return total, (2 if total >= 7.5 else 1), f"ML:{ml_pts:.1f}|Espaço:{espaco_pts:.1f}|Fluxo:{liq_pts:.1f}({detalhes})"

    async def scan_market(self):
        # A primeira chamada ao universe_provider acontece aqui dentro agora
        ativos = await universe_provider.get_ativos()
        logging.info(f"📡 Iniciando ciclo: Varrendo {len(ativos)} moedas...")

        semaphore = asyncio.Semaphore(15)

        async def process_asset(symbol, nlp_score):
            async with semaphore:
                return await self.analyze_symbol(symbol, nlp_score)

        try:
            while True:
                try:
                    # Garantimos que a lista de moedas seja consultada de forma assíncrona a cada ciclo
                    ativos = await universe_provider.get_ativos()
                    
                    if not ativos:
                        logging.warning("⚠️ Universo vazio. Aguardando recuperação do provedor...")
                        await asyncio.sleep(5)
                        continue

                    max_open = getattr(Config, 'MAX_OPEN_TRADES', 5)
                    open_trades = await self.db.get_open_trades_count()
                    
                    if open_trades >= max_open:
                        await asyncio.sleep(60)
                        continue

                    nlp_score = await self.news.get_sentiment_score()
                    
                    tasks = [process_asset(s, nlp_score) for s in ativos]
                    results = await asyncio.gather(*tasks)

                    cycle_opportunities = [r for r in results if r is not None]

                    min_score = getattr(Config, 'MIN_SCORE_ENTRY', 5.2)
                    espera_minutos = getattr(Config, 'TEMPO_ESPERA_HOLD_MINUTOS', 5)
                    
                    if cycle_opportunities:
                        all_sorted = sorted(cycle_opportunities, key=lambda x: (x['score'], x['expected_move'], x['prob']), reverse=True)
                        top_opps = [opp for opp in all_sorted if opp['score'] >= min_score][:20]

                        if top_opps:
                            await self.db.update_elite_signals(top_opps)
                            for opp in top_opps:
                                logging.info(f"🔥 SINAL DETECTADO ({opp['symbol']}): {opp['direction']} | Score: {opp['score']:.1f}/10 | {opp['reasoning']}")
                                self.next_trade_time[opp['symbol']] = time.time() + (espera_minutos * 60)
                            for opp in all_sorted:
                                if opp['score'] < min_score:
                                    self.next_trade_time[opp['symbol']] = time.time() + (espera_minutos * 60)
                        else:
                            await self.db.update_elite_signals([])
                            for opp in all_sorted:
                                self.next_trade_time[opp['symbol']] = time.time() + (espera_minutos * 60)
                    else:
                        await self.db.update_elite_signals([])

                    ciclo_segundos = getattr(Config, 'CICLO_SEGUNDOS', 5)
                    await asyncio.sleep(ciclo_segundos)
                except Exception as e:
                    logging.error(f"Erro crítico no loop principal do radar: {e}")
                    await asyncio.sleep(5)
        finally:
            await self.shutdown()

if __name__ == "__main__":
    radar = RadarCore()
    try:
        asyncio.run(radar.scan_market())
    except KeyboardInterrupt:
        logging.info("Encerrando bot com segurança...")
    except Exception as e:
        logging.error(f"Erro na execução principal: {e}")
