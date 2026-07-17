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
            self.btc_train_cache = self.fetch_historical_sync(ativos[0], Config.CANDLES_TREINAMENTO_ML)
            self.btc_train_time = time.time()
        return self.btc_train_cache

    def fetch_historical_sync(self, symbol, limit):
        try:
            # 🛡️ FIX INSTITUCIONAL: Acelerador controlado para respeitar a Bybit
            time.sleep(1.5) 
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, Config.TIMEFRAME, limit=limit)
            
            # Extração de Open Interest (Contratos Abertos)
            try:
                time.sleep(0.5) 
                oi_data = self.exchange.fetch_open_interest_history(symbol, Config.TIMEFRAME, limit=limit)
                oi_map = {int(item.get('timestamp', 0)): float(item.get('openInterestValue') or item.get('info', {}).get('openInterest', 0)) for item in oi_data}
            except Exception:
                oi_map = {}

            # Extração de Funding Rate (Taxa de Financiamento - Caça à Liquidez)
            try:
                time.sleep(0.5)
                fr_data = self.exchange.fetch_funding_rate_history(symbol, limit=limit)
                fr_map = {int(item.get('timestamp', 0)): float(item.get('fundingRate', 0)) for item in fr_data}
            except Exception:
                fr_map = {}

            merged = []
            last_oi = 0.0
            last_fr = 0.0
            for bar in ohlcv:
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
        
        # Criação segura do EMA_20, caso falte da biblioteca customizada
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
        # 🚀 INÍCIO DOS 15 SENSORES QUANTITATIVOS INSTITUCIONAIS
        # ====================================================================================

        # 1. Divergência OBV vs. RSI (Volume vs Força Relativa)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_slope'] = df['obv'].diff(3).fillna(0.0)
        df['obv_rsi_div'] = np.where((df['obv_slope'] > 0) & (df['rsi_slope'] < 0), 1.0,
                            np.where((df['obv_slope'] < 0) & (df['rsi_slope'] > 0), -1.0, 0.0))

        # 2. Aceleração do CVD (Segunda Derivada do Delta de Volume)
        df['cvd_accel'] = df['cvd'].diff(3).diff(3).fillna(0.0)

        # 3. Distância da VWAP Ancorada (Preço Justo Diário)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vol_price'] = df['volume'] * df['typical_price']
        df['date_only'] = df['datetime'].dt.date
        df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-9)
        df['vwap_dist'] = ((df['close'] - df['vwap']) / (df['vwap'] + 1e-9)).fillna(0.0)

        # 4. Vácuo de Liquidez (Dinâmica do Spread Proxy)
        df['liq_vacuum'] = ((df['high'] - df['low']) / (df['volume'] + 1e-9)).rolling(10).mean().fillna(0.0)

        # 5. Variação do Funding Rate (Ataque ao Varejo)
        if 'funding_rate' in df.columns:
            df['funding_rate'] = df['funding_rate'].replace(0, np.nan).ffill().fillna(0.0)
            df['funding_rate_delta'] = df['funding_rate'].diff(3).fillna(0.0)
        else:
            df['funding_rate_delta'] = 0.0

        # 6. Momentum de Open Interest (Velocidade de Contratos)
        if 'open_interest' in df.columns:
            df['oi_momentum'] = (df['open_interest'].diff(5) / (df['open_interest'].rolling(20).mean() + 1e-9)).fillna(0.0)
        else:
            df['oi_momentum'] = 0.0

        # 7. Smart Money Proxy (Negative Volume Index - NVI)
        vol_drop = df['volume'] < df['volume'].shift(1)
        roc = df['close'].pct_change().fillna(0.0)
        df['nvi'] = (vol_drop * roc).cumsum().fillna(0.0)

        # 8. Expoente de Hurst (Mapeamento de Regimes: Tendência vs Lateral)
        lag1_var = df['log_return'].rolling(20).var().fillna(0.0)
        lag5_var = df['close'].pct_change(5).rolling(20).var().fillna(0.0)
        df['hurst_proxy'] = (np.log(lag5_var + 1e-9) / np.log(lag1_var + 1e-9)).fillna(0.0)

        # 9. Z-Score (Extremos Matemáticos do Preço)
        df['z_score_50'] = ((df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std() + 1e-9)).fillna(0.0)

        # 10. Autocorrelação de Retornos (Memória do Mercado)
        df['autocorr_3'] = df['log_return'].rolling(20).apply(lambda x: x.autocorr(lag=3) if len(x.dropna()) > 3 else 0, raw=False).fillna(0.0)

        # 11. Dimensão Fractal (Caos vs Estrutura)
        n_period = 20
        high_max = df['high'].rolling(n_period).max()
        low_min = df['low'].rolling(n_period).min()
        path_length = np.abs(df['close'].diff()).rolling(n_period).sum()
        df['fractal_dim'] = (np.log(path_length + 1e-9) / np.log((high_max - low_min) + 1e-9)).fillna(0.0)

        # 12. Squeeze Keltner/Bollinger (O Efeito Mola)
        atr_14 = df.get('ATRr_14', df['close'].rolling(14).std())
        df['kc_upper'] = df['EMA_20'] + (1.5 * atr_14)
        df['kc_lower'] = df['EMA_20'] - (1.5 * atr_14)
        bb_width = bbu - bbl
        kc_width = df['kc_upper'] - df['kc_lower']
        df['squeeze_ratio'] = (bb_width / (kc_width + 1e-9)).fillna(1.0)

        # 13. ATR Normalizado (Combustível Absoluto)
        df['natr'] = ((atr_14 / df['close']) * 100).fillna(0.0)

        # 14. Skewness de Retornos (Assimetria do Medo/Ganância)
        df['skewness_20'] = df['log_return'].rolling(20).skew().fillna(0.0)

        # 15. Velocidade do Histograma MACD
        ema9_macd = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['MACD_12_26_9'] - ema9_macd
        df['macd_hist_vel'] = df['macd_hist'].diff(2).fillna(0.0)

        # Limpeza rápida das colunas extras geradas
        df.drop(columns=['typical_price', 'vol_price', 'date_only', 'cum_vol_price', 'cum_vol', 'vwap', 'obv', 'obv_slope', 'kc_upper', 'kc_lower', 'macd_hist'], inplace=True, errors='ignore')

        # ====================================================================================
        # FIM DA INJEÇÃO DOS 15 SENSORES QUANTITATIVOS
        # ====================================================================================

        df_indexed = df.set_index('datetime')

        df_1h = df_indexed['close'].resample('1h').last().to_frame(name='close_1h')
        df_1h.ffill(inplace=True)
        df_1h['ema_20_1h'] = df_1h['close_1h'].ewm(span=20, adjust=False).mean()

        df_4h = df_indexed['close'].resample('4h').last().to_frame(name='close_4h')
        df_4h.ffill(inplace=True)
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
                    targets[i] = 1
                    break
                elif lows[i + j] <= sl: 
                    targets[i] = 2
                    break
            if np.isnan(targets[i]):
                targets[i] = 0 
                
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

        hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42, min_covar=1e-3)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm_model.fit(df[['log_return', 'volatility_cluster']])
            df['hmm_regime'] = hmm_model.predict(df[['log_return', 'volatility_cluster']])
        except Exception:
            hmm_model = DummyHMM() 
            df['hmm_regime'] = 0

        # INJEÇÃO DAS NOVAS COLUNAS NA MATRIZ DE APRENDIZADO
        features = [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 
            'btc_correlation', 'noise_index',
            # --- Novos 15 Sensores ---
            'obv_rsi_div', 'cvd_accel', 'vwap_dist', 'liq_vacuum', 'funding_rate_delta', 
            'oi_momentum', 'nvi', 'hurst_proxy', 'z_score_50', 'autocorr_3', 
            'fractal_dim', 'squeeze_ratio', 'natr', 'skewness_20', 'macd_hist_vel'
        ]
        X, y = df[features], df['target'].astype(int)

        lgbm = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='mlogloss', n_jobs=-1)
        cb_model = CatBoostClassifier(iterations=120, learning_rate=0.05, depth=5, silent=True, random_state=42, thread_count=-1)

        lgbm.fit(X, y)
        xgb_model.fit(X, y)
        cb_model.fit(X, y)

        X_meta = np.column_stack([np.asarray(lgbm.predict_proba(X)), np.asarray(xgb_model.predict_proba(X)), np.asarray(cb_model.predict_proba(X))])
        
        meta_learner = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
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

        return f"✅ Cérebro de {symbol} treinado e salvo com sucesso."

    def iniciar_ciclo_treinamento(self):
        ativos = Config.get_ativos()
        logging.info(f"🚀 Iniciando Treinador Quantitativo 10/10 (Lote de {len(ativos)} moedas)...")
        
        btc_data_raw = self.get_btc_data_sync()
        btc_train_df = pd.DataFrame(btc_data_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']) if btc_data_raw else None

        # 🛡️ FIX INSTITUCIONAL: Limita a 2 processos simultâneos para evitar banimento da Bybit
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

        logging.info(f"💤 Treinamento concluído. O motor vai hibernar por {Config.HORAS_RETREINO} horas.")

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
            
