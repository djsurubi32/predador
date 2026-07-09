import sys
import os
import asyncio
import math
import logging
import time
import json
import aiosqlite
import joblib
import warnings
from concurrent.futures import ProcessPoolExecutor

import requests
from requests.adapters import HTTPAdapter
import ccxt
import pandas as pd
import numpy as np
import websockets

from config import Config
from ta_indicators import add_custom_ta

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore")

# Memória global por processo para cache de modelos (evita I/O repetido dentro do processo filho)
_PROCESS_MODEL_CACHE = {}

def _worker_predict_with_brain(symbol: str, models_dir: str, current_data_row: dict) -> tuple:
    """
    Função global de nível superior executada estritamente dentro dos processos filhos
    do ProcessPoolExecutor. Isola completamente o processamento matemático do GIL principal.
    """
    try:
        safe_name = symbol.replace('/', '_').replace(':', '_')
        path = os.path.join(models_dir, f"{safe_name}.pkl")
        if not os.path.exists(path):
            return "Erro", 0.0

        file_mod_time = os.path.getmtime(path)

        # Gerenciamento de Cache interno do Processo Filho
        if symbol not in _PROCESS_MODEL_CACHE or _PROCESS_MODEL_CACHE[symbol]['mod_time'] < file_mod_time:
            if len(_PROCESS_MODEL_CACHE) >= 15:
                _PROCESS_MODEL_CACHE.clear()
            brain = joblib.load(path)
            _PROCESS_MODEL_CACHE[symbol] = {'brain': brain, 'mod_time': file_mod_time}

        brain = _PROCESS_MODEL_CACHE[symbol]['brain']

        features = ['RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 'btc_correlation', 'noise_index']

        X_pred = pd.DataFrame([{f: current_data_row.get(f, 0.0) for f in features}])

        if hasattr(brain['hmm'], 'predict'):
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
        prob_queda = final_probs[2] * 100 if len(final_probs) > 1 else 0

        if prob_alta >= 60.0:
            return "ALTA", prob_alta
        elif prob_queda >= 60.0:
            return "QUEDA", prob_queda
        else:
            return "Inconclusivo", max(prob_alta, prob_queda)

    except Exception:
        return "Erro", 0.0

class Database:
    def __init__(self, db_name="predador_v31.db"):
        self.db_name = db_name

    async def inicializar(self):
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            await conn.execute('PRAGMA journal_mode=WAL;')
            await conn.execute('''CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, entry REAL,
                sl REAL, tp REAL, qty REAL, force INTEGER, open_time REAL)''')
            await conn.execute('''CREATE TABLE IF NOT EXISTS elite_signals (
                symbol TEXT PRIMARY KEY, direction TEXT, price REAL, prob REAL,
                score REAL, atr REAL, sma_atr REAL, funding REAL, reasoning TEXT,
                timestamp REAL)''')
            await conn.commit()

    async def update_elite_signals(self, signals):
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            await conn.execute('DELETE FROM elite_signals')
            now = time.time()
            for sig in signals:
                await conn.execute('''INSERT INTO elite_signals
                    (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (sig['symbol'], sig['direction'], sig['price'], sig['prob'],
                     sig['score'], sig['current_atr'], sig['sma_atr'], sig['funding'],
                     sig['reasoning'], now))
            await conn.commit()

    async def get_open_trades_count(self):
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            async with conn.execute('SELECT COUNT(*) FROM trades') as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

class BybitWebSocketManager:
    def __init__(self, ccxt_symbols):
        self.url = "wss://stream.bybit.com/v5/public/linear"
        self.ccxt_symbols = ccxt_symbols
        self.ticker_cache = {}
        self.ccxt_to_bybit = {}
        self.bybit_to_ccxt = {}
        self._build_mappings()

    def _build_mappings(self):
        for sym in self.ccxt_symbols:
            native = sym.split(':')[0].replace('/', '')
            self.ccxt_to_bybit[sym] = native
            self.bybit_to_ccxt[native] = sym
            self.ticker_cache[sym] = {"last": 0.0, "bid": 0.0, "ask": 0.0}

    async def start(self):
        asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            try:
                logging.info(f"🌐 Conectando ao WebSocket de Redes da Bybit V5 Public...")
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    args = [f"tickers.{self.ccxt_to_bybit[sym]}" for sym in self.ccxt_symbols]
                    sub_msg = {"op": "subscribe", "args": args}
                    await ws.send(json.dumps(sub_msg))
                    logging.info(f"✅ WebSocket conectado e inscrito em {len(args)} fluxos de ticks ativos.")

                    while True:
                        res = await ws.recv()
                        data = json.loads(res)

                        if "topic" in data and "data" in data:
                            topic = data["topic"]
                            native_symbol = topic.replace("tickers.", "")
                            ccxt_symbol = self.bybit_to_ccxt.get(native_symbol)

                            if ccxt_symbol:
                                tick_info = data["data"]
                                if "lastPrice" in tick_info and tick_info["lastPrice"]:
                                    self.ticker_cache[ccxt_symbol]["last"] = float(tick_info["lastPrice"])
                                if "bid1Price" in tick_info and tick_info["bid1Price"]:
                                    self.ticker_cache[ccxt_symbol]["bid"] = float(tick_info["bid1Price"])
                                if "ask1Price" in tick_info and tick_info["ask1Price"]:
                                    self.ticker_cache[ccxt_symbol]["ask"] = float(tick_info["ask1Price"])

            except Exception as e:
                logging.error(f"❌ Falha ou desconexão no fluxo contínuo do WebSocket: {e}. Reconectando em 5 segundos...")
                await asyncio.sleep(5)

    def get_ticker(self, symbol):
        return self.ticker_cache.get(symbol, {"last": 0.0, "bid": 0.0, "ask": 0.0})

class LiquidityCore:
    def __init__(self):
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
                ask_vol = sum(vol * math.exp(-100 * abs(price - mid_price) / mid_price) for price, vol in ob['asks'][:10])
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
        self.last_fetch = 0
        self.current_sentiment = 0.0

    def fetch_and_score_sync(self):
        return 0.0

    async def get_sentiment_score(self):
        return 0.0

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
        self.next_trade_time = {}
        self.ws_manager = BybitWebSocketManager(Config.get_ativos())

        # ⚙️ Pool de Multiprocessamento dedicado para o isolamento do GIL
        self.process_pool = ProcessPoolExecutor(max_workers=4)

    def prepare_features(self, df, btc_df=None):
        df = add_custom_ta(df)

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema12 - ema26
        df['SMA_ATR_100'] = df['ATRr_14'].rolling(window=100).mean() if 'ATRr_14' in df.columns else 0.0

        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.ffill(inplace=True); df.fillna(0.0, inplace=True)

        ema5 = df['close'].ewm(span=5, adjust=False).mean()
        df['close_smooth'] = ema5.ewm(span=5, adjust=False).mean()

        df['noise_index'] = (abs(df['close'] - df['close_smooth']) / (df['close'] + 1e-9)).fillna(0.0)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1).replace(0, 1e-9))
        df['volatility_cluster'] = df['log_return'].rolling(window=20).std()
        df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-9)
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
        df['oi_change'], df['oi_trend'], df['oi_price_divergence'] = 0.0, 0.0, 0.0

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df_indexed = df.set_index('datetime')
        df_1h = df_indexed['close'].resample('1h').last().to_frame(name='close_1h').ffill()
        df_1h['ema_20_1h'] = df_1h['close_1h'].ewm(span=20, adjust=False).mean()
        df_4h = df_indexed['close'].resample('4h').last().to_frame(name='close_4h').ffill()
        df_4h['ema_20_4h'] = df_4h['close_4h'].ewm(span=20, adjust=False).mean()

        df_indexed = df_indexed.join(df_1h[['ema_20_1h']], how='left').ffill()
        df_indexed = df_indexed.join(df_4h[['ema_20_4h']], how='left').ffill()
        df_indexed.reset_index(drop=True, inplace=True); df = df_indexed
        df['ema_20_1h'] = df['ema_20_1h'].fillna(df['close']); df['ema_20_4h'] = df['ema_20_4h'].fillna(df['close'])
        df['mtf_dist_1h'] = (df['close'] - df['ema_20_1h']) / df['ema_20_1h']; df['mtf_dist_4h'] = (df['close'] - df['ema_20_4h']) / df['ema_20_4h']

        df['btc_log_return'], df['btc_correlation'] = df['log_return'], 1.0
        df.fillna(0.0, inplace=True)
        return df

    async def analyze_symbol(self, symbol, news_sentiment):
        if time.time() < self.next_trade_time.get(symbol, 0): return None

        try:
            ohlcv = await asyncio.to_thread(self.public_exchange.fetch_ohlcv, symbol, Config.TIMEFRAME, limit=400)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.prepare_features(df, None)
            if len(df) < 30: return None

            ws_tick = self.ws_manager.get_ticker(symbol)
            raw_price = ws_tick["last"] if ws_tick["last"] > 0 else df.iloc[-1]['close']

            current_atr = float(df.iloc[-1].get('ATRr_14', raw_price * 0.005))
            sma_atr = float(df.iloc[-1].get('SMA_ATR_100', current_atr))
            expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0

            # 🚀 EXECUÇÃO MULTIPROCESSADA: Despacha inferência pesada para o ProcessPoolExecutor
            loop = asyncio.get_running_loop()
            current_row_dict = df.iloc[-1].to_dict()

            ml_text, ml_prob = await loop.run_in_executor(
                self.process_pool,
                _worker_predict_with_brain,
                symbol,
                Config.MODELS_DIR,
                current_row_dict
            )

            direction = 'BUY' if "ALTA" in ml_text else 'SELL' if "QUEDA" in ml_text else 'HOLD'
            if direction == 'HOLD': return None

            vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict = await self.liquidity.get_liquidity_report(symbol)
            score, force, reasoning = self.calculate_conviction_score(direction, ml_prob, vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict, news_sentiment, current_atr, raw_price)

            return {
                'symbol': symbol, 'direction': direction, 'price': float(raw_price),
                'prob': ml_prob, 'score': score, 'current_atr': current_atr,
                'sma_atr': sma_atr, 'funding': 0.0, 'reasoning': reasoning,
                'expected_move': expected_move_pct
            }
        except Exception:
            return None

    def calculate_conviction_score(self, direction, ml_prob, vwap_dict, cvd_dict, spoof_buy_dict, spoof_sell_dict, nlp_sentiment, current_atr, raw_price):
        ml_pts = 0.0
        if ml_prob >= 95.0: ml_pts = 5.0
        elif ml_prob >= 90.0: ml_pts = 4.0
        else: return 0, 1, f"VETO ML: Probabilidade Abaixo de 90% ({ml_prob:.1f}%)."

        expected_move_pct = (current_atr * 2.0 / raw_price) * 100.0
        espaco_pts = 0.0
        if expected_move_pct >= 2.0: espaco_pts = 3.0
        elif expected_move_pct >= 1.0: espaco_pts = 2.0
        elif expected_move_pct >= 0.5: espaco_pts = 1.0
        else: return 0, 1, f"VETO ESPAÇO: Alvo muito curto ({expected_move_pct:.2f}%)."

        liq_pts = 0.0
        apoios = []
        if direction == 'BUY' and vwap_dict.get('Binance', 50) >= 50.5: liq_pts += 1.0; apoios.append('Binance_VWAP')
        elif direction == 'SELL' and vwap_dict.get('Binance', 50) <= 49.5: liq_pts += 1.0; apoios.append('Binance_VWAP')

        if direction == 'BUY' and vwap_dict.get('Bybit', 50) >= 50.5: liq_pts += 1.0; apoios.append('Bybit_VWAP')
        elif direction == 'SELL' and vwap_dict.get('Bybit', 50) <= 49.5: liq_pts += 1.0; apoios.append('Bybit_VWAP')

        cvd_buy = cvd_dict.get('Binance', 50.0) >= 50.5 or cvd_dict.get('Bybit', 50.0) >= 50.5
        cvd_sell = cvd_dict.get('Binance', 50.0) <= 49.5 or cvd_dict.get('Bybit', 50.0) <= 49.5
        if direction == 'BUY' and cvd_buy: liq_pts += 1.0; apoios.append('CVD')
        if direction == 'SELL' and cvd_sell: liq_pts += 1.0; apoios.append('CVD')

        if liq_pts < 1.0: return 0, 1, "VETO VWAP: Sem apoio de liquidez."

        spoof_buy = spoof_buy_dict.get('Binance', False) or spoof_buy_dict.get('Bybit', False)
        spoof_sell = spoof_sell_dict.get('Binance', False) or spoof_sell_dict.get('Bybit', False)
        if direction == 'BUY' and spoof_buy: return 0, 1, "VETO SPOOFING."
        if direction == 'SELL' and spoof_sell: return 0, 1, "VETO SPOOFING."

        total = ml_pts + espaco_pts + liq_pts
        detalhes = ", ".join(apoios) if apoios else "Nenhum"
        motivo_detalhado = f"ML_Dir:{ml_pts:.1f} | ML_Espaço:{espaco_pts:.1f}({expected_move_pct:.1f}%) | Fluxo:{liq_pts:.1f}({detalhes})"
        return total, (2 if total >= 8.0 else 1), motivo_detalhado

    async def scan_market(self):
        logging.info(f"📡 Radar inicializado com Isolamento Multiprocessado (GIL-Free).")
        await self.db.inicializar()
        await self.ws_manager.start()

        while True:
            try:
                ativos = Config.get_ativos()
                open_trades = await self.db.get_open_trades_count()
                vagas_disponiveis = Config.MAX_OPEN_TRADES - open_trades

                if vagas_disponiveis <= 0:
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
                    all_sorted = sorted(cycle_opportunities, key=lambda x: (x['prob'], x['score'], x['expected_move']), reverse=True)
                    valid_opps = [opp for opp in all_sorted if opp['score'] >= Config.MIN_SCORE_ENTRY]
                    top_opps = valid_opps[:vagas_disponiveis]

                    if top_opps:
                        await self.db.update_elite_signals(top_opps)
                        logging.info(f"🏆 Mesa do Leilão updated com as {len(top_opps)} MELHORES oportunidades (Vagas: {vagas_disponiveis}).")

                        for opp in top_opps:
                            logging.info(f"🔥 SINAL RANKING 1º ESCALÃO ({opp['symbol']}): {opp['direction']} | Score: {opp['score']:.1f}/10 | Prob: {opp['prob']:.1f}% | {opp['reasoning']}")
                            self.next_trade_time[opp['symbol']] = time.time() + (60 * 60)

                        for opp in all_sorted:
                            if opp not in top_opps:
                                self.next_trade_time[opp['symbol']] = time.time() + (Config.TEMPO_ESPERA_HOLD_MINUTOS * 60)
                    else:
                        await self.db.update_elite_signals([])
                        logging.info(f"♻️ Varredura concluída. Nenhuma superou a nota de corte ({Config.MIN_SCORE_ENTRY}/10) e Probabilidade > 90%.")
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
