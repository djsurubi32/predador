import os
import time
import logging
import warnings
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
import ccxt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from config import Config
from ta_indicators import add_custom_ta

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [TREINADOR] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

class DummyHMM:
    def predict(self, X):
        return np.zeros(len(X))

class MotorTreinamento:
    def __init__(self):
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        opcoes_ccxt = {
            'enableRateLimit': True, 
            'options': {'defaultType': 'swap'},
            'session': session
        }
            
        self.exchange = ccxt.bybit(opcoes_ccxt)
        self.btc_train_cache = []
        self.btc_train_time = 0

    def get_btc_data_sync(self):
        ativos = Config.get_ativos()
        if not ativos: return []
        
        if time.time() - self.btc_train_time > 3600 or not self.btc_train_cache:
            logging.info("Sincronizando benchmark temporal (BTC) com paginação profunda...")
            self.btc_train_cache = self.fetch_historical_sync(ativos[0], Config.CANDLES_TREINAMENTO_ML)
            self.btc_train_time = time.time()
        return self.btc_train_cache

    def fetch_historical_sync(self, symbol, limit):
        """
        🚀 OTIMIZAÇÃO: Motor de Paginação Profunda. 
        Burla o limite de 1000 velas da Bybit e força o download da matriz completa de treinamento.
        """
        try:
            all_ohlcv = []
            # Calcula o timestamp de início baseado no limite e timeframe de 15m
            since_ms = int((time.time() - (limit * 15 * 60)) * 1000)
            
            while len(all_ohlcv) < limit:
                time.sleep(0.2) # Acelerador respeitoso (Rate Limit)
                batch = self.exchange.fetch_ohlcv(symbol, Config.TIMEFRAME, since=since_ms, limit=1000)
                if not batch or len(batch) == 0:
                    break
                
                since_ms = batch[-1][0] + 1
                all_ohlcv.extend(batch)
            
            all_ohlcv = all_ohlcv[-limit:] # Garante o corte exato

            # Extração de Open Interest com Paginação Simplificada
            try:
                time.sleep(0.5) 
                oi_data = self.exchange.fetch_open_interest_history(symbol, Config.TIMEFRAME, limit=1000)
                oi_map = {int(item.get('timestamp', 0)): float(item.get('openInterestValue') or item.get('info', {}).get('openInterest', 0)) for item in oi_data}
            except Exception:
                oi_map = {}

            # Extração de Funding Rate
            try:
                time.sleep(0.5)
                fr_data = self.exchange.fetch_funding_rate_history(symbol, limit=1000)
                fr_map = {int(item.get('timestamp', 0)): float(item.get('fundingRate', 0)) for item in fr_data}
            except Exception:
                fr_map = {}

            merged = []
            last_oi = 0.0
            last_fr = 0.0
            for bar in all_ohlcv:
                ts = int(bar[0])
                if oi_map.get(ts, 0.0) != 0.0:
                    last_oi = oi_map.get(ts, 0.0)
                if fr_map.get(ts) is not None:
                    last_fr = fr_map.get(ts)
                
                merged.append([bar[0], bar[1], bar[2], bar[3], bar[4], bar[5], last_oi, last_fr])
            return merged
        except Exception as e:
            logging.error(f"Erro de I/O ao extrair histórico de {symbol}: {e}")
            return []

    def prepare_features(self, df, btc_df=None):
        df = add_custom_ta(df)

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema12 - ema26

        if 'ATRr_14' in df.columns:
            df['SMA_ATR_100'] = df['ATRr_14'].rolling(window=100).mean()
        else:
            df['SMA_ATR_100'] = 0.0

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
        # INJEÇÃO DOS 15 SENSORES QUANTITATIVOS INSTITUCIONAIS
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

        if btc_df is not None and not btc_df.empty:
            btc_temp = btc_df[['timestamp', 'close']].copy()
            btc_temp.rename(columns={'close': 'btc_close'}, inplace=True)
            df = pd.merge(df, btc_temp, on='timestamp', how='left')
            df['btc_close'] = df['btc_close'].ffill().bfill()

            df['btc_log_return'] = np.log(df['btc_close'] / df['btc_close'].shift(1).replace(0, 1e-9)).fillna(0.0)
            df['btc_correlation'] = df['log_return'].rolling(window=20).corr(df['btc_log_return']).fillna(0.0)
            df.drop(columns=['btc_close'], inplace=True)
        else:
            df['btc_log_return'] = df['log_return']
            df['btc_correlation'] = 1.0

        df.fillna(0.0, inplace=True)
        return df

    def apply_triple_barrier(self, df):
        horizon, tp_pct, sl_pct = Config.BARRIER_HORIZON, Config.BARRIER_TP_PCT, Config.BARRIER_SL_PCT
        targets = np.full(len(df), np.nan) 
        closes, highs, lows = df['close'].values, df['high'].values, df['low'].values

        for i in range(len(df) - horizon):
            entry, tp, sl = closes[i], closes[i] * tp_pct, closes[i] * sl_pct
            for j in range(1, horizon + 1):
                if highs[i + j] >= tp: 
                    targets[i] = 1 # Rompimento Comprador (BUY)
                    break
                elif lows[i + j] <= sl: 
                    targets[i] = 2 # Rompimento Vendedor (SELL)
                    break
            if np.isnan(targets[i]):
                targets[i] = 0 # Indefinição / Tempo esgotado
                
        df['target'] = targets
        return df

    def processar_moeda(self, symbol, btc_train_df):
        bars = self.fetch_historical_sync(symbol, Config.CANDLES_TREINAMENTO_ML)
        if len(bars) < 100:
            return f"⚠️ Dados insuficientes para {symbol}."

        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.ffill(inplace=True)
        df = self.prepare_features(df, btc_train_df)
        df = self.apply_triple_barrier(df)
        
        df = df.dropna(subset=['target']).copy()

        if len(df) < 50: 
            return f"⚠️ Alvos insuficientes após purga em {symbol}."

        df['target'] = df['target'].astype(int)
        
        # Garante a existência das 3 classes para o algoritmo funcionar bidirecionalmente
        classes_presentes = set(df['target'].unique())
        classes_necessarias = {0, 1, 2}
        classes_faltantes = classes_necessarias - classes_presentes
        if classes_faltantes:
            linhas_dummy = []
            for c in classes_faltantes:
                linha = df.iloc[-1:].copy()
                linha['target'] = int(c)
                linhas_dummy.append(linha)
            df = pd.concat([df] + linhas_dummy, ignore_index=True)
            df['target'] = df['target'].astype(int)

        # Treinamento do HMM (Regimes de Mercado)
        hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42, min_covar=1e-3)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm_model.fit(df[['log_return', 'volatility_cluster']])
            df['hmm_regime'] = hmm_model.predict(df[['log_return', 'volatility_cluster']])
        except Exception:
            hmm_model = DummyHMM() 
            df['hmm_regime'] = 0

        features = [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 
            'btc_correlation', 'noise_index',
            'obv_rsi_div', 'cvd_accel', 'vwap_dist', 'liq_vacuum', 'funding_rate_delta', 
            'oi_momentum', 'nvi', 'hurst_proxy', 'z_score_50', 'autocorr_3', 
            'fractal_dim', 'squeeze_ratio', 'natr', 'skewness_20', 'macd_hist_vel'
        ]
        X, y = df[features], df['target']

        # 🚀 OTIMIZAÇÃO: Time-Series Split (Hold-out Validation)
        # Proteção absoluta contra overfitting e decoragem de mercado passado.
        split_idx = int(len(X) * 0.8) # Treina com 80%, valida com 20%
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 🚀 OTIMIZAÇÃO: Hiperparâmetros Institucionais (L1/L2 Regularization e Balanceamento)
        lgbm = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=6, num_leaves=31, 
            class_weight='balanced', # Resolve o desbalanceamento quantitativo
            reg_alpha=0.1, reg_lambda=0.1, # Trava o Overfitting
            random_state=42, verbose=-1, n_jobs=-1
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=5, 
            early_stopping_rounds=15, # Para se não melhorar na validação
            random_state=42, eval_metric='mlogloss', n_jobs=-1
        )
        
        cb_model = CatBoostClassifier(
            iterations=150, learning_rate=0.03, depth=5, 
            auto_class_weights='Balanced', # Força foco em Longs e Shorts
            l2_leaf_reg=3.0, early_stopping_rounds=15,
            silent=True, random_state=42, thread_count=-1
        )

        # Treinamento com Early Stopping
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # Matriz de Meta-Features extraída com as predições cegas do conjunto completo
        X_meta = np.column_stack([np.asarray(lgbm.predict_proba(X)), np.asarray(xgb_model.predict_proba(X)), np.asarray(cb_model.predict_proba(X))])
        
        # O Meta-Learner ganha "class_weight='balanced'" para não viciar no Neutro (0)
        meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
        meta_learner.fit(X_meta, y)

        brain_data = {
            'hmm': hmm_model,
            'lgbm': lgbm,
            'xgb': xgb_model,
            'catboost': cb_model,
            'meta': meta_learner,
            'last_trained': int(time.time())
        }

        safe_symbol_name = symbol.replace('/', '_').replace(':', '_')
        file_path = os.path.join(Config.MODELS_DIR, f"{safe_symbol_name}.pkl")
        joblib.dump(brain_data, file_path)

        return f"✅ Cérebro de {symbol} forjado com alta precisão (Paginação e Anti-Overfitting aplicados)."

    def iniciar_ciclo_treinamento(self):
        ativos = Config.get_ativos()
        logging.info(f"🚀 Iniciando Forja Institucional (Lote de {len(ativos)} moedas)...")
        
        btc_data_raw = self.get_btc_data_sync()
        btc_train_df = pd.DataFrame(btc_data_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']) if btc_data_raw else None

        # Limita processos para não fuzilar a memória e sofrer IP Ban
        max_threads = 2 
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futuros = {executor.submit(self.processar_moeda, symbol, btc_train_df): symbol for symbol in ativos}
            
            for future in as_completed(futuros):
                symbol = futuros[future]
                try:
                    resultado = future.result()
                    if "⚠️" in resultado:
                        logging.warning(resultado)
                    else:
                        logging.info(resultado)
                except Exception as exc:
                    logging.error(f"❌ Falha fatal no worker processando {symbol}: {exc}")

        logging.info(f"💤 Treinamento blindado concluído. O motor vai hibernar por {Config.HORAS_RETREINO} horas.")

def main():
    treinador = MotorTreinamento()
    while True:
        try:
            treinador.iniciar_ciclo_treinamento()
            time.sleep(Config.HORAS_RETREINO * 3600)
        except Exception as e:
            logging.error(f"Erro Crítico no Loop do Treinador: {e}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("🛑 Treinador encerrado pelo usuário.")
