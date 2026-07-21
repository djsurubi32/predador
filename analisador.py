import sys
import os
import asyncio
import math
import logging
import time
import sqlite3
import joblib
import warnings

import requests
from requests.adapters import HTTPAdapter
import ccxt
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
        self.db_name = db_name
        self._create_tables()

    def _create_tables(self):
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

    def _update_elite_signals_sync(self, signals):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM elite_signals')
            now = time.time()
            for sig in signals:
                cursor.execute('''INSERT INTO elite_signals
                    (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (sig['symbol'], sig['direction'], sig['price'], sig['prob'],
                     sig['score'], sig['current_atr'], sig['sma_atr'], sig['funding'],
                     sig['reasoning'], now))
            conn.commit()

    async def update_elite_signals(self, signals):
        await asyncio.to_thread(self._update_elite_signals_sync, signals)

    def _get_open_trades_count_sync(self):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM trades')
            return cursor.fetchone()[0]

    async def get_open_trades_count(self):
        return await asyncio.to_thread(self._get_open_trades_count_sync)

class LiquidityCore:
    def __init__(self):
        # 🚀 EXPANSÃO DA RODOVIA DE REDE
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        self.exchanges = {
            "Binance": ccxt.binance({'enableRateLimit': True, 'session': session}),
            "Bybit": ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'session': session}),
        }
        self.ob_history = {name: {} for name in self.exchanges.keys()}

    def fetch_liquidity_data_sync(self, exchange, ex_name, symbol_spot):
        try:
            ob = exchange.fetch_order_book(symbol_spot, limit=20)
            vwap_imb, bid_vol, ask_vol = 50.0, 0.0, 0.0

            if ob.get('bids') and ob.get('asks'):
                mid_price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2.0
                bid_vol = sum(vol * math.exp(-100 * abs(price - mid_price) / mid_price) for price, vol in ob['bids'][:10])
                ask_vol = sum(vol * math.exp(-100 * abs(price - mid_price) / mid_price) for price, vol in ob['bids'][:10])
                total = bid_vol + ask_vol
                if total > 0: vwap_imb = (bid_vol / total) * 100

            spoof_buy, spoof_sell = False, False
            prev = self.ob_history[ex_name].get(symbol_spot)
            if prev:
                if prev['bids'] > 0 and (prev['bids'] - bid_vol) / prev['bids'] > 0.20: spoof_buy = True
                if prev['asks'] > 0 and (prev['asks'] - ask_vol) / prev['asks'] > 0.20: spoof_sell = True

            self.ob_history[ex_name][symbol_spot] = {'bids': bid_vol, 'asks': ask_vol}

            cvd_imb = 50.0
            try:
                trades = exchange.fetch_trades(symbol_spot, limit=200)
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
        tasks = [asyncio.to_thread(self.fetch_liquidity_data_sync, ex, name, symbol_spot) for name, ex in self.exchanges.items()]
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
            self.model = SentenceTransformer(Config.NLP_MODEL_NAME)
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
        if not titles: return 0.0
        try:
            embs = self.model.encode(titles)
            bull = np.dot(embs, self.bull_emb.T) / (norm(embs, axis=1, keepdims=True) * norm(self.bull_emb))
            bear = np.dot(embs, self.bear_emb.T) / (norm(embs, axis=1, keepdims=True) * norm(self.bear_emb))
            return float(np.mean(bull - bear))
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
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        self.public_exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'session': session})
        self.db = Database()
        self.liquidity = LiquidityCore()
        self.news = LocalNewsCore()
        self.next_trade_time = {ativo: 0 for ativo in Config.get_ativos()}
        self.loaded_models = {}
        self.MAX_MODELS_IN_RAM = 30

    def load_brain(self, symbol):
        safe_name = symbol.replace('/', '_').replace(':', '_')
        path = os.path.join(Config.MODELS_DIR, f"{safe_name}.pkl")
        if not os.path.exists(path): return None

        file_mod_time = os.path.getmtime(path)
        if symbol not in self.loaded_models or self.loaded_models[symbol]['mod_time'] < file_mod_time:
            try:
                if len(self.loaded_models) >= self.MAX_MODELS_IN_RAM:
                    self.loaded_models.clear()
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
        
        if 'ATRr_14' in df.columns:
            df['SMA_ATR_100'] = df['ATRr_14'].rolling(window=100).mean()
        else:
            df['SMA_ATR_100'] = 0.0

        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)

        ema5 = df['close'].ewm(span=5, adjust=False).mean()
        df['close_smooth'] = ema5.ewm(span=5, adjust=False).mean()

        df['noise_index'] = (abs(df['close'] - df['close_smooth']) / (df['close'] + 1e-9)).fillna(0.0)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1).replace(0, 1e-9))
        df['volatility_cluster'] = df['log_return'].rolling(window=20).std()
        
        vol_mean, vol_std = df['volume'].rolling(20).mean(), df['volume'].rolling(20).std()
        df['vol_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
        
        if 'EMA_20' not in df.columns:
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['price_vs_ema'] = df['close'] - df['EMA_20']
        
        bbl = df.get('BBL_20_2.0', df['close'])
        bbu = df.get('BBU_20_2.0', df['close'])
        df['bb_pos'] = (df['close'] - bbl) / (bbu - bbl + 1e-9)
        
        df['rsi_slope'] = df.get('RSI_14', pd.Series(0, index=df.index)).diff(3)
        df['price_slope'] = df['close_smooth'].diff(3).fillna(0.0)
        df['rsi_divergence'] = np.where((df['price_slope'] < 0) & (df['rsi_slope'] > 0), 1, np.where((df['price_slope'] > 0) & (df['rsi_slope'] < 0), -1, 0))
        
        df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
        df['cvd'] = (df['volume'] * df['candle_dir']).cumsum()
        df['cvd_trend'] = df['cvd'] - df['cvd'].rolling(20).mean()

        # Tratamento seguro para sensores não-presentes ao vivo (OI e Funding)
        if 'open_interest' in df.columns:
            df['oi_temp'] = df['open_interest'].replace(0, np.nan).ffill().bfill()
            df['oi_change'] = df['oi_temp'].pct_change(fill_method=None).fillna(0.0)
            df['oi_trend'] = df['oi_change'].rolling(window=5).mean().fillna(0.0)
            df['price_pct'] = df['close'].pct_change(fill_method=None).fillna(0.0)
            df['oi_price_divergence'] = np.where((df['price_pct'] > 0) & (df['oi_change'] > 0), 1.0,
                                        np.where((df['price_pct'] < 0) & (df['oi_change'] > 0), -1.0,
                                        np.where((df['price_pct'] > 0) & (df['oi_change'] < 0), -0.5,
                                        np.where((df['price_pct'] < 0) & (df['oi_change'] < 0), 0.5, 0.0))))
            df.drop(columns=['oi_temp', 'price_pct'], inplace=True)
        else:
            df['oi_change'] = 0.0; df['oi_trend'] = 0.0; df['oi_price_divergence'] = 0.0

        # ====================================================================================
        # INJEÇÃO DOS 15 NOVOS SENSORES QUANTITATIVOS AO VIVO (34 Features Totais)
        # ====================================================================================
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_slope'] = df['obv'].diff(3).fillna(0.0)
        df['obv_rsi_div'] = np.where((df['obv_slope'] > 0) & (df['rsi_slope'] < 0), 1.0,
                            np.where((df['obv_slope'] < 0) & (df['rsi_slope'] > 0), -1.0, 0.0))

        df['cvd_accel'] = df['cvd'].diff(3).diff(3).fillna(0.0)

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vol_price'] = df['volume'] * df['typical_price']
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

        if 'open_interest' in df.columns:
            df['oi_momentum'] = (df['open_interest'].diff(5) / (df['open_interest'].rolling(20).mean() + 1e-9)).fillna(0.0)
        else:
            df['oi_momentum'] = 0.0

        vol_drop = df['volume'] < df['volume'].shift(1)
        roc = df['close'].pct_change().fillna(0.0)
        df['nvi'] = (vol_drop * roc).cumsum().fillna(0.0)

        lag1_var = df['log_return'].rolling(20).var().fillna(0.0)
        lag5_var = df['close'].pct_change(5).rolling(20).var().fillna(0.0)
        df['hurst_proxy'] = (np.log(lag5_var + 1e-9) / np.log(lag1_var + 1e-9)).fillna(0.0)

        df['z_score_50'] = ((df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std() + 1e-9)).fillna(0.0)
        df['autocorr_3'] = df['log_return'].rolling(20).apply(lambda x: x.autocorr(lag=3) if len(x.dropna()) > 3 else 0, raw=False).fillna(0.0)

        n_period = 20
        high_max = df['high'].rolling(n_period).max()
        low_min = df['low'].rolling(n_period).min()
        path_length = np.abs(df['close'].diff()).rolling(n_period).sum()
        df['fractal_dim'] = (np.log(path_length + 1e-9) / np.log((high_max - low_min) + 1e-9)).fillna(0.0)

        atr_14 = df.get('ATRr_14', df['close'].rolling(14).std())
        df['kc_upper'] = df['EMA_20'] + (1.5 * atr_14)
        df['kc_lower'] = df['EMA_20'] - (1.5 * atr_14)
        bb_width = bbu - bbl
        kc_width = df['kc_upper'] - df['kc_lower']
        df['squeeze_ratio'] = (bb_width / (kc_width + 1e-9)).fillna(1.0)

        df['natr'] = ((atr_14 / df['close']) * 100).fillna(0.0)
        df['skewness_20'] = df['log_return'].rolling(20).skew().fillna(0.0)

        ema9_macd = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['MACD_12_26_9'] - ema9_macd
        df['macd_hist_vel'] = df['macd_hist'].diff(2).fillna(0.0)

        df.drop(columns=['typical_price', 'vol_price', 'date_only', 'cum_vol_price', 'cum_vol', 'vwap', 'obv', 'obv_slope', 'kc_upper', 'kc_lower', 'macd_hist'], inplace=True, errors='ignore')

        # MTF e Alinhamento Temporal
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
        # 🛡️ FIX INSTITUCIONAL: A Matriz de 34 Sensores Completa e Perfeita
        features = [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 
            'btc_correlation', 'noise_index',
            'obv_rsi_div', 'cvd_accel', 'vwap_dist', 'liq_vacuum', 'funding_rate_delta', 
            'oi_momentum', 'nvi', 'hurst_proxy', 'z_score_50', 'autocorr_3', 
            'fractal_dim', 'squeeze_ratio', 'natr', 'skewness_20', 'macd_hist_vel'
        ]
        
        try:
            X_pred = pd.DataFrame([{f: current_data_row.get(f, 0.0) for f in features}])
            if hasattr(brain['hmm'], 'predict'):
                try: X_pred['hmm_regime'] = brain['hmm'].predict(X_pred[['log_return', 'volatility_cluster']])
                except: X_pred['hmm_regime'] = 0
            else:
                X_pred['hmm_regime'] = 0

            X_meta = np.column_stack((brain['lgbm'].predict_proba(X_pred), brain['xgb'].predict_proba(X_pred), brain['catboost'].predict_proba(X_pred)))
            final_probs = brain['meta'].predict_proba(X_meta)[0]

            # 🛡️ CORREÇÃO DO BUG BIDIRECIONAL: Indexação Segura (Sem IndexErrors em classe 2)
            prob_alta = final_probs[1] * 100 if len(final_probs) > 1 else 0
            prob_queda = final_probs[2] * 100 if len(final_probs) > 2 else 0

            if prob_alta >= 60.0: return "ALTA", prob_alta
            elif prob_queda >= 60.0: return "QUEDA", prob_queda
            else: return "Inconclusivo", max(prob_alta, prob_queda)
        except Exception as e:
            logging.error(f"Erro no predict_with_brain: {e}")
            return "Erro", 0.0

    async def analyze_symbol(self, symbol, news_sentiment):
        if time.time() < self.next_trade_time.get(symbol, 0): return None
        brain = self.load_brain(symbol)
        if not brain: return None

        try:
            ohlcv = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, symbol, Config.TIMEFRAME, limit=400)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.prepare_features(df, None)
            if len(df) < 30: return None

            raw_price = df.iloc[-1]['close']
            current_atr = float(df.iloc[-1].get('ATRr_14', raw_price * 0.005))
            sma_atr = float(df.iloc[-1].get('SMA_ATR_100', current_atr))
            expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0

            # Extração da Posição nas Bandas (ML_Espaço)
            bb_pos = float(df.iloc[-1].get('bb_pos', 0.5))

            ml_text, ml_prob = self.predict_with_brain(brain, df.iloc[-1].to_dict())
            direction = 'BUY' if "ALTA" in ml_text else 'SELL' if "QUEDA" in ml_text else 'HOLD'
            if direction == 'HOLD': return None

            vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict = await self.liquidity.get_liquidity_report(symbol)
            
            # Passando bb_pos para a trava rígida de conviction score
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
            return None

    def calculate_conviction_score(self, direction, ml_prob, vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict, nlp_sentiment, current_atr, raw_price, bb_pos):
        
        # --- 🛡️ BLOQUEIO RÍGIDO INSTITUCIONAL (ARMADILHA DO ESPAÇO) ---
        # bb_pos >= 0.95 significa exaustão compradora extrema (Topo).
        # bb_pos <= 0.05 significa exaustão vendedora extrema (Fundo).
        if direction == 'BUY' and bb_pos >= 0.95:
            return 0, 1, f"VETO ESPAÇO RIGIDO: Compra de Topo Bloqueada (bb_pos={bb_pos:.2f}). Risco extremo de reversão."
        if direction == 'SELL' and bb_pos <= 0.05:
            return 0, 1, f"VETO ESPAÇO RIGIDO: Venda de Fundo Bloqueada (bb_pos={bb_pos:.2f}). Risco extremo de repique."
        
        ml_pts = 0.0
        if ml_prob >= 80.0: ml_pts = 4.0
        elif ml_prob >= 70.0: ml_pts = 3.0
        elif ml_prob >= 65.0: ml_pts = 2.0
        elif ml_prob >= 60.0: ml_pts = 1.0
        else: return 0, 1, f"VETO ML: Probabilidade Baixa ({ml_prob:.1f}%)."
        
        expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0
        espaco_pts = 0.0
        if expected_move_pct >= 2.0: espaco_pts = 3.0
        elif expected_move_pct >= 1.0: espaco_pts = 2.0
        elif expected_move_pct >= 0.5: espaco_pts = 1.0
        else: return 0, 1, f"VETO ESPAÇO: Alvo muito curto ({expected_move_pct:.2f}%)."
        
        liq_pts = 0.0
        apoios = []
        if direction == 'BUY' and vwap_dict.get('Binance', 50) >= 50.5: 
            liq_pts += 1.0
            apoios.append('Binance_VWAP')
        elif direction == 'SELL' and vwap_dict.get('Binance', 50) <= 49.5: 
            liq_pts += 1.0
            apoios.append('Binance_VWAP')

        if direction == 'BUY' and vwap_dict.get('Bybit', 50) >= 50.5: 
            liq_pts += 1.0
            apoios.append('Bybit_VWAP')
        elif direction == 'SELL' and vwap_dict.get('Bybit', 50) <= 49.5: 
            liq_pts += 1.0
            apoios.append('Bybit_VWAP')

        cvd_buy = cvd_dict.get('Binance', 50.0) >= 50.5 or cvd_dict.get('Bybit', 50.0) >= 50.5
        cvd_sell = cvd_dict.get('Binance', 50.0) <= 49.5 or cvd_dict.get('Bybit', 50.0) <= 49.5
        if direction == 'BUY' and cvd_buy: 
            liq_pts += 1.0
            apoios.append('CVD')
        if direction == 'SELL' and cvd_sell: 
            liq_pts += 1.0
            apoios.append('CVD')

        if liq_pts < 1.0: return 0, 1, "VETO VWAP: Sem apoio de liquidez institucional."
        
        spoof_buy = spoof_buy_dict.get('Binance', False) or spoof_buy_dict.get('Bybit', False)
        spoof_sell = spoof_sell_dict.get('Binance', False) or spoof_sell_dict.get('Bybit', False)
        if direction == 'BUY' and spoof_buy: return 0, 1, "VETO SPOOFING."
        if direction == 'SELL' and spoof_sell: return 0, 1, "VETO SPOOFING."
        
        total = ml_pts + espaco_pts + liq_pts
        detalhes = ", ".join(apoios) if apoios else "Nenhum"
        motivo_detalhado = f"ML_Dir:{ml_pts:.1f} | ML_Espaço:{espaco_pts:.1f}({expected_move_pct:.1f}%) | Fluxo:{liq_pts:.1f}({detalhes})"
        return total, (2 if total >= 8.0 else 1), motivo_detalhado

    async def scan_market(self):
        ativos = Config.get_ativos()
        logging.info(f"📡 Varrendo {len(ativos)} moedas na Nova Escala (0 a 10)...")
        while True:
            try:
                open_trades = await self.db.get_open_trades_count()
                if open_trades >= Config.MAX_OPEN_TRADES:
                    logging.info("⏸️ Balde global cheio. Radar em espera.")
                    await asyncio.sleep(60)
                    continue

                nlp_score = await self.news.get_sentiment_score()
                cycle_opportunities = []
                for s in ativos:
                    signal = await self.analyze_symbol(s, nlp_score)
                    if signal: cycle_opportunities.append(signal)
                    await asyncio.sleep(0.1)

                if cycle_opportunities:
                    all_sorted = sorted(cycle_opportunities, key=lambda x: (x['score'], x['expected_move'], x['prob']), reverse=True)
                    top_opps = [opp for opp in all_sorted if opp['score'] >= Config.MIN_SCORE_ENTRY][:20]

                    if top_opps:
                        await self.db.update_elite_signals(top_opps)
                        logging.info(f"🏆 Mesa do Leilão atualizada com {len(top_opps)} oportunidades.")
                        for opp in top_opps:
                            logging.info(f"🔥 SINAL DETETADO ({opp['symbol']}): {opp['direction']} | Score: {opp['score']:.1f}/10 | {opp['reasoning']}")
                            self.next_trade_time[opp['symbol']] = time.time() + (Config.TEMPO_ESPERA_HOLD_MINUTOS * 60)
                        for opp in all_sorted:
                            if opp['score'] < Config.MIN_SCORE_ENTRY:
                                self.next_trade_time[opp['symbol']] = time.time() + (Config.TEMPO_ESPERA_HOLD_MINUTOS * 60)
                    else:
                        await self.db.update_elite_signals([])
                        best_5 = all_sorted[:5]
                        relatorio = f"♻️ Varredura concluída. Nenhuma atingiu o corte ({Config.MIN_SCORE_ENTRY}/10).\n"
                        for i, opp in enumerate(best_5, 1):
                            relatorio += f"   {i}º {opp['symbol']} ({opp['direction']}) | Score: {opp['score']:.1f} | Motivo: {opp['reasoning']}\n"
                        logging.info(relatorio.strip())
                        for opp in all_sorted:
                            self.next_trade_time[opp['symbol']] = time.time() + (Config.TEMPO_ESPERA_HOLD_MINUTOS * 60)
                else:
                    await self.db.update_elite_signals([])
                    logging.info(f"♻️ Varredura concluída. Mercado em total indefinição.")

                await asyncio.sleep(Config.CICLO_SEGUNDOS)
            except Exception as e:
                logging.error(f"Erro no loop do radar: {e}")
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(RadarCore().scan_market())
    except KeyboardInterrupt:
        logging.info("🛑 Analisador encerrado pelo usuário.")
