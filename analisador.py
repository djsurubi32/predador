import os
import time
import logging
import sqlite3
import asyncio
import numpy as np
import pandas as pd
import ccxt
import joblib

from config import Config
from treinador import MotorTreinamento

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

class AnalisadorRadar:
    def __init__(self, exchange: ccxt.bybit, db_name: str = "predador_v31.db"):
        self.exchange = exchange
        self.db_name = db_name
        self.motor = MotorTreinamento() # Herda a matemática complexa de 19 colunas do treinador
        self.btc_cache = None
        self.btc_cache_time = 0

    async def obter_btc_data(self):
        agora = time.time()
        if not self.btc_cache or (agora - self.btc_cache_time) > 300:
            bars = await asyncio.to_thread(self.motor.fetch_historical_sync, 'BTC/USDT:USDT', 150)
            if bars:
                self.btc_cache = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
                self.btc_cache_time = agora
        return self.btc_cache

    async def analisar_ativo(self, symbol: str):
        model_name = symbol.replace('/', '_').replace(':', '_')
        model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            return

        try:
            bars = await asyncio.to_thread(self.motor.fetch_historical_sync, symbol, 150)
            if not bars or len(bars) < 50: return

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
            btc_df = await self.obter_btc_data()

            if btc_df is None or btc_df.empty: return

            # A MÁGICA: Prepara os dados exatos exigidos pelo Meta-Learner (Fim do erro de incompatibilidade)
            df = self.motor.prepare_features(df, btc_df)
            latest_row = df.iloc[-1].copy()

            features = ['RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 'btc_correlation', 'noise_index']

            input_data = pd.DataFrame([latest_row[features]], columns=features)
            if input_data.isna().any().any(): return

            brain = joblib.load(model_path)
            hmm = brain['hmm']
            lgbm = brain['lgbm']
            xgb = brain['xgb']
            cat = brain['catboost']
            meta = brain['meta']

            # Alimenta e calcula o modelo HMM isoladamente
            try:
                input_data['hmm_regime'] = hmm.predict(input_data[['log_return', 'volatility_cluster']])
            except:
                input_data['hmm_regime'] = 0

            # Previsão Ensemble: Cruzamento do LigehtGBM, XGBoost e CatBoost
            X_meta = np.column_stack([
                np.asarray(lgbm.predict_proba(input_data)),
                np.asarray(xgb.predict_proba(input_data)),
                np.asarray(cat.predict_proba(input_data))
            ])

            prob_up = float(meta.predict_proba(X_meta)[0][1])

            direction = None
            prob_final = 0.0

            if prob_up >= 0.70:
                direction = 'BUY'
                prob_final = prob_up
            elif prob_up <= 0.30:
                direction = 'SELL'
                prob_final = 1.0 - prob_up

            if direction is not None:
                score = round(prob_final * 10, 1)

                if score >= Config.MIN_SCORE_ENTRY:

                    mtf_1h = latest_row['mtf_dist_1h']
                    mtf_4h = latest_row['mtf_dist_4h']
                    trend_1h = 'Bull' if mtf_1h > 0 else 'Bear'
                    trend_4h = 'Bull' if mtf_4h > 0 else 'Bear'

                    rsi_atual = latest_row['RSI_14']
                    if rsi_atual > 65 or rsi_atual < 35: fluxo_score = 3.0
                    elif rsi_atual > 55 or rsi_atual < 45: fluxo_score = 2.0
                    else: fluxo_score = 1.0

                    dev_vwap = latest_row.get('bb_pos', 0) * 100
                    abs_dev = abs(dev_vwap)
                    if abs_dev < 10: espaco_score = 3.0
                    elif abs_dev < 30: espaco_score = 2.0
                    else: espaco_score = 1.0

                    reasoning = (
                        f"ML_Dir:{score:.1f} | "
                        f"ML_Espaço:{espaco_score:.1f}({dev_vwap:.1f}%) | "
                        f"Fluxo:{fluxo_score:.1f} | "
                        f"Macro1h:{trend_1h} | "
                        f"Macro4h:{trend_4h}"
                    )

                    self.salvar_sinal_db(
                        symbol=symbol,
                        direction=direction,
                        price=float(latest_row['close']),
                        prob=prob_final * 100,
                        score=score,
                        atr=float(latest_row.get('ATRr_14', 0)),
                        sma_atr=float(latest_row.get('SMA_ATR_100', 0)),
                        reasoning=reasoning
                    )
        except Exception as e:
            logging.error(f"Erro ao processar predição de {symbol}: {e}")

    def salvar_sinal_db(self, symbol, direction, price, prob, score, atr, sma_atr, reasoning):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO elite_signals
                (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (symbol, direction, price, prob, score, atr, sma_atr, 0.0, reasoning, time.time()))
            conn.commit()

    async def limpar_sinais_db(self):
        try:
            with sqlite3.connect(self.db_name, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM elite_signals')
                conn.commit()
        except:
            pass

    async def loop_radar(self):
        logging.info("📡 Radar Ensemble inicializado (Fila Sequencial Anti-Ban).")
        while True:
            try:
                await self.limpar_sinais_db()
                ativos = Config.get_ativos()

                for symbol in ativos:
                    await self.analisar_ativo(symbol)
                    await asyncio.sleep(1.5)
            except Exception as e:
                logging.error(f"Erro no loop principal do radar: {e}")
            await asyncio.sleep(60)
