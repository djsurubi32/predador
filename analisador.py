import sys
import os
import asyncio
import math
import logging
import time
import sqlite3
import joblib
import warnings

# Otimização: aiosqlite para chamadas assíncronas nativas de banco
import aiosqlite
# Otimização: CCXT Assíncrono nativo, dispensando requests/HTTPAdapter
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from numpy.linalg import norm
import feedparser
from sentence_transformers import SentenceTransformer

from config import Config
from ta_indicators import add_custom_ta

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore")

class Database:
    def __init__(self, db_name="predador_v31.db"):
        self.db_name = getattr(Config, 'DB_NAME', db_name)
        self._create_tables_sync()

    def _create_tables_sync(self):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, entry REAL,
                sl REAL, tp REAL, qty REAL, force INTEGER, open_time REAL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS elite_signals (
                symbol TEXT PRIMARY KEY, direction TEXT, price REAL, prob REAL,
                score REAL, atr REAL, sma_atr REAL, funding REAL, reasoning TEXT,
                timestamp REAL)''')
            conn.commit()

    async def update_elite_signals(self, signals):
        now = time.time()
        data_to_insert = [
            (sig['symbol'], sig['direction'], sig['price'], sig['prob'],
             sig['score'], sig['current_atr'], sig['sma_atr'], sig['funding'],
             sig['reasoning'], now) for sig in signals
        ]

        async with aiosqlite.connect(self.db_name, timeout=30) as db:
            await db.execute('DELETE FROM elite_signals')
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
            "Binance": ccxt_async.binance({'enableRateLimit': True}),
            "Bybit": ccxt_async.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}}),
        }
        self.ob_history = {name: {} for name in self.exchanges.keys()}

    async def fetch_liquidity_data(self, exchange, ex_name, symbol_spot):
        try:
            ob = await exchange.fetch_order_book(symbol_spot, limit=20)
            vwap_imb, bid_vol, ask_vol = 50.0, 0.0, 0.0

            if ob.get('bids') and ob.get('asks'):
                mid_price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2.0
                bid_vol = sum(vol * math.exp(-50 * abs(price - mid_price) / mid_price) for price, vol in ob['bids'][:10])
                ask_vol = sum(vol * math.exp(-50 * abs(price - mid_price) / mid_price) for price, vol in ob['asks'][:10])
                total = bid_vol + ask_vol
                if total > 0: vwap_imb = (bid_vol / total) * 100

            spoof_buy, spoof_sell = False, False
            prev = self.ob_history[ex_name].get(symbol_spot)
            if prev:
                if prev['bids'] > 0 and (prev['bids'] - bid_vol) / prev['bids'] > 0.35: spoof_buy = True
                if prev['asks'] > 0 and (prev['asks'] - ask_vol) / prev['asks'] > 0.35: spoof_sell = True

            self.ob_history[ex_name][symbol_spot] = {'bids': bid_vol, 'asks': ask_vol}

            cvd_imb = 50.0
            try:
                trades = await exchange.fetch_trades(symbol_spot, limit=200)
                if trades:
                    cvd_buy = sum(t.get('amount', 0) for t in trades if t.get('side') == 'buy')
                    cvd_sell = sum(t.get('amount', 0) for t in trades if t.get('side') == 'sell')
                    if (cvd_buy + cvd_sell) > 0: cvd_imb = (cvd_buy / (cvd_buy + cvd_sell)) * 100
            except Exception:
                pass

            return vwap_imb, cvd_imb, spoof_buy, spoof_sell
        except Exception:
            return 50.0, 50.0, False, False

    async def get_liquidity_report(self, symbol):
        symbol_spot = symbol.split(':')[0]
        tasks = [self.fetch_liquidity_data(ex, name, symbol_spot) for name, ex in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for r in results:
            if isinstance(r, Exception) or not isinstance(r, tuple):
                valid_results.append((50.0, 50.0, False, False))
            else:
                valid_results.append(r)

        return {name: r[0] for name, r in zip(self.exchanges.keys(), valid_results)}, \
               {name: r[1] for name, r in zip(self.exchanges.keys(), valid_results)}, \
               {name: r[2] for name, r in zip(self.exchanges.keys(), valid_results)}, \
               {name: r[3] for name, r in zip(self.exchanges.keys(), valid_results)}

class LocalNewsCore:
    def __init__(self):
        try:
            model_name = getattr(Config, 'NLP_MODEL_NAME', 'all-MiniLM-L6-v2')
            self.model = SentenceTransformer(model_name)
            self.bull_emb = self.model.encode(["bull market positive good news surge adoption"])
            self.bear_emb = self.model.encode(["bear market crash negative bad news regulation"])
        except Exception:
            self.model = None
        self.last_fetch = 0
        self.current_sentiment = 0.0

    def fetch_and_score_sync(self):
        if not self.model: return 0.0
        titles = []
        for url in ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]:
            try:
                feed = feedparser.parse(url)
                titles.extend([entry.title for entry in feed.entries[:8]])
            except Exception:
                pass
        
        if not titles: 
            return 0.0
            
        try:
            embs = self.model.encode(titles)
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
        self.next_trade_time = {ativo: 0 for ativo in Config.get_ativos()}
        self.loaded_models = {}
        # Parametrizado
        self.MAX_MODELS_IN_RAM = getattr(Config, 'MAX_MODELS_IN_RAM', 30)

    def load_brain(self, symbol):
        safe_name = symbol.replace('/', '_').replace(':', '_')
        models_dir = getattr(Config, 'MODELS_DIR', 'modelos')
        path = os.path.join(models_dir, f"{safe_name}.pkl")
        
        if not os.path.exists(path): 
            return None

        file_mod_time = os.path.getmtime(path)
        if symbol not in self.loaded_models or self.loaded_models[symbol]['mod_time'] < file_mod_time:
            try:
                if len(self.loaded_models) >= self.MAX_MODELS_IN_RAM:
                    oldest = min(self.loaded_models, key=lambda k: self.loaded_models[k]['mod_time'])
                    del self.loaded_models[oldest]
                    logging.info(f"Limpando RAM: Modelo {oldest} removido.")
                
                brain = joblib.load(path)
                self.loaded_models[symbol] = {'brain': brain, 'mod_time': file_mod_time}
            except Exception:
                return None
        return self.loaded_models[symbol]['brain']

    def prepare_features(self, df, btc_df=None):
        df = add_custom_ta(df)

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema12 - ema26

        df['SMA_ATR_100'] = df['ATRr_14'].rolling(window=100).mean() if 'ATRr_14' in df.columns else 0.0

        numeric_cols = df.columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)

        ema5 = df['close'].ewm(span=5, adjust=False).mean()
        df['close_smooth'] = ema5.ewm(span=5, adjust=False).mean()

        df['noise_index'] = (abs(df['close'] - df['close_smooth']) / (df['close'] + 1e-9)).fillna(0.0)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1).replace(0, 1e-9))
        df['volatility_cluster'] = df['log_return'].rolling(window=20).std()

        vol_roll = df['volume'].rolling(20)
        df['vol_zscore'] = (df['volume'] - vol_roll.mean()) / (vol_roll.std() + 1e-9)

        if 'EMA_20' not in df.columns:
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['price_vs_ema'] = df['close'] - df['EMA_20']

        bbl = df.get('BBL_20_2.0', df['close'])
        bbu = df.get('BBU_20_2.0', df['close'])
        df['bb_pos'] = (df['close'] - bbl) / (bbu - bbl + 1e-9)

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
            df['oi_momentum'] = (df['open_interest'].diff(5) / (df['open_interest'].rolling(20).mean() + 1e-9)).fillna(0.0)
        else:
            df['oi_change'], df['oi_trend'], df['oi_price_divergence'], df['oi_momentum'] = 0.0, 0.0, 0.0, 0.0

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_slope'] = df['obv'].diff(3).fillna(0.0)
        df['obv_rsi_div'] = np.where((df['obv_slope'] > 0) & (df['rsi_slope'] < 0), 1.0,
                            np.where((df['obv_slope'] < 0) & (df['rsi_slope'] > 0), -1.0, 0.0))

        df['cvd_accel'] = df['cvd'].diff(3).diff(3).fillna(0.0)

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vol_price'] = df['volume'] * typical_price
        df['date_only'] = df['datetime'].dt.date
        df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-9)
        df['vwap_dist'] = ((df['close'] - df['vwap']) / (df['vwap'] + 1e-9)).fillna(0.0)

        df['liq_vacuum'] = ((df['high'] - df['low']) / (df['volume'] + 1e-9)).rolling(10).mean().fillna(0.0)

        if 'funding_rate' in df.columns:
            df['funding_rate'] = df['funding_rate'].replace(0, np.nan).ffill().fillna(0.0)
            df['funding_rate_delta'] = df['funding_rate'].diff(3).fillna(0.0)
        else:
            df['funding_rate_delta'] = 0.0

        vol_drop = df['volume'] < df['volume'].shift(1)
        roc = df['close'].pct_change().fillna(0.0)
        df['nvi'] = (vol_drop * roc).cumsum().fillna(0.0)

        lag1_var = df['log_return'].rolling(20).var().fillna(0.0)
        lag5_var = df['close'].pct_change(5).rolling(20).var().fillna(0.0)
        df['hurst_proxy'] = (np.log(lag5_var + 1e-9) / np.log(lag1_var + 1e-9)).fillna(0.0)

        roll50 = df['close'].rolling(50)
        df['z_score_50'] = ((df['close'] - roll50.mean()) / (roll50.std() + 1e-9)).fillna(0.0)

        df['autocorr_3'] = df['log_return'].rolling(20).apply(lambda x: x.autocorr(lag=3) if len(x) >= 4 else 0, raw=False).fillna(0.0)

        n_period = 20
        high_max = df['high'].rolling(n_period).max()
        low_min = df['low'].rolling(n_period).min()
        path_length = np.abs(df['close'].diff()).rolling(n_period).sum()
        df['fractal_dim'] = (np.log(path_length + 1e-9) / np.log((high_max - low_min) + 1e-9)).fillna(0.0)

        atr_14 = df.get('ATRr_14', df['close'].rolling(14).std())
        kc_upper = df['EMA_20'] + (1.5 * atr_14)
        kc_lower = df['EMA_20'] - (1.5 * atr_14)
        df['squeeze_ratio'] = ((bbu - bbl) / (kc_upper - kc_lower + 1e-9)).fillna(1.0)

        df['natr'] = ((atr_14 / df['close']) * 100).fillna(0.0)
        df['skewness_20'] = df['log_return'].rolling(20).skew().fillna(0.0)

        ema9_macd = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['macd_hist_vel'] = (df['MACD_12_26_9'] - ema9_macd).diff(2).fillna(0.0)

        cols_to_drop = ['vol_price', 'date_only', 'cum_vol_price', 'cum_vol', 'vwap', 'obv', 'obv_slope']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        df_indexed = df.set_index('datetime')
        df_1h = df_indexed['close'].resample('1h').last().to_frame(name='close_1h').ffill()
        df_1h['ema_20_1h'] = df_1h['close_1h'].ewm(span=20, adjust=False).mean()
        df_4h = df_indexed['close'].resample('4h').last().to_frame(name='close_4h').ffill()
        df_4h['ema_20_4h'] = df_4h['close_4h'].ewm(span=20, adjust=False).mean()

        df_indexed = df_indexed.join(df_1h[['ema_20_1h']], how='left').ffill()
        df_indexed = df_indexed.join(df_4h[['ema_20_4h']], how='left').ffill()
        df_indexed.reset_index(drop=True, inplace=True)
        df = df_indexed

        df['ema_20_1h'] = df['ema_20_1h'].fillna(df['close'])
        df['ema_20_4h'] = df['ema_20_4h'].fillna(df['close'])
        df['mtf_dist_1h'] = (df['close'] - df['ema_20_1h']) / df['ema_20_1h']
        df['mtf_dist_4h'] = (df['close'] - df['ema_20_4h']) / df['ema_20_4h']

        df['btc_log_return'], df['btc_correlation'] = df['log_return'], 1.0
        df.fillna(0.0, inplace=True)
        return df

    def predict_with_brain(self, brain, current_data_row):
        features = [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14',
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change',
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return',
            'btc_correlation', 'noise_index', 'obv_rsi_div', 'cvd_accel', 'vwap_dist',
            'liq_vacuum', 'funding_rate_delta', 'oi_momentum', 'nvi', 'hurst_proxy',
            'z_score_50', 'autocorr_3', 'fractal_dim', 'squeeze_ratio', 'natr',
            'skewness_20', 'macd_hist_vel'
        ]

        try:
            X_pred = pd.DataFrame([{f: current_data_row.get(f, 0.0) for f in features}])
            if hasattr(brain.get('hmm'), 'predict'):
                try: 
                    X_pred['hmm_regime'] = brain['hmm'].predict(X_pred[['log_return', 'volatility_cluster']])
                except: 
                    X_pred['hmm_regime'] = 0
            else:
                X_pred['hmm_regime'] = 0

            X_meta = np.column_stack((
                brain['lgbm'].predict_proba(X_pred),
                brain['xgb'].predict_proba(X_pred),
                brain['catboost'].predict_proba(X_pred)
            ))
            final_probs = brain['meta'].predict_proba(X_meta)[0]

            prob_alta = final_probs[1] * 100 if len(final_probs) > 1 else 0
            prob_queda = final_probs[2] * 100 if len(final_probs) > 2 else 0

            # CORREÇÃO 1: Utilizando o limite do config dinâmico
            min_prob_predict = getattr(Config, 'MIN_PROB_PREDICT', 52.0)

            if prob_alta >= min_prob_predict: 
                return "ALTA", prob_alta, prob_queda
            elif prob_queda >= min_prob_predict: 
                return "QUEDA", prob_alta, prob_queda
            else: 
                return "HOLD", prob_alta, prob_queda
        except Exception:
            return "Erro", 0.0, 0.0

    async def analyze_symbol(self, symbol, news_sentiment):
        espera_hold = getattr(Config, 'TEMPO_ESPERA_HOLD_MINUTOS', 5)
        if time.time() < self.next_trade_time.get(symbol, 0): 
            return None
            
        brain = self.load_brain(symbol)
        if not brain: 
            return None

        try:
            timeframe = getattr(Config, 'TIMEFRAME', '5m')
            ohlcv = await self.public_exchange.fetch_ohlcv(symbol, timeframe, limit=400)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df = self.prepare_features(df, None)
            if len(df) < 30: 
                return None

            raw_price = df.iloc[-1]['close']
            current_atr = float(df.iloc[-1].get('ATRr_14', raw_price * 0.005))
            sma_atr = float(df.iloc[-1].get('SMA_ATR_100', current_atr))
            expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0

            bb_pos = float(df.iloc[-1].get('bb_pos', 0.5))

            ml_text, prob_alta, prob_queda = self.predict_with_brain(brain, df.iloc[-1].to_dict())
            
            direction = 'BUY' if ml_text == "ALTA" else 'SELL' if ml_text == "QUEDA" else 'HOLD'
            ml_prob = max(prob_alta, prob_queda)
            
            if direction == 'HOLD': 
                logging.info(f"[{symbol}] HOLD - ML Inconclusivo (Alta: {prob_alta:.2f}% | Queda: {prob_queda:.2f}%)")
                return None

            vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict = await self.liquidity.get_liquidity_report(symbol)

            score, force, reasoning = self.calculate_conviction_score(
                direction, ml_prob, vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict,
                news_sentiment, current_atr, raw_price, bb_pos
            )

            return {
                'symbol': symbol, 'direction': direction, 'price': float(raw_price),
                'prob': ml_prob, 'score': score, 'current_atr': current_atr,
                'sma_atr': sma_atr, 'funding': 0.0, 'reasoning': reasoning,
                'expected_move': expected_move_pct
            }
        except Exception:
    