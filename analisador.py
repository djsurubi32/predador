import os
import time
import logging
import sqlite3
import threading
import asyncio
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import ccxt
import joblib

from config import Config
from treinador import MotorTreinamento # Importa o verdadeiro Cérebro Institucional que você construiu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

class AnalisadorRadar:
    def __init__(self, exchange: ccxt.bybit, db_name: str = "predador_v31.db"):
        self.exchange = exchange
        self.db_name = db_name
        self.motor = MotorTreinamento() # Instancia a máquina pesada de 19 colunas
        self.btc_cache = None
        self.btc_cache_time = 0

    async def obter_btc_data(self):
        agora = time.time()
        # Cache de 5 minutos para o Bitcoin para não sobrecarregar a API da Bybit
        if not self.btc_cache or (agora - self.btc_cache_time) > 300:
            bars = await asyncio.to_thread(self.motor.fetch_historical_sync, 'BTC/USDT:USDT', 150)
            if bars:
                self.btc_cache = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
                self.btc_cache_time = agora
        return self.btc_cache

    async def analisar_ativo(self, symbol: str):
        # Proteção de Nomes
        model_name = symbol.replace('/', '_').replace(':', '_')
        model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            return

        try:
            # 1. Puxa os dados brutos com a pausa de segurança do treinador
            bars = await asyncio.to_thread(self.motor.fetch_historical_sync, symbol, 150)
            if not bars or len(bars) < 50:
                return

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])

            # 2. Puxa a bússola macro do BTC
            btc_df = await self.obter_btc_data()
            if btc_df is None or btc_df.empty:
                return

            # 3. A MÁGICA: Gera as 19 colunas exatas exigidas pelo seu Meta-Learner
            df = self.motor.prepare_features(df, btc_df)
            latest_row = df.iloc[-1].copy()

            features = [
                'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos',
                'ADX_14', 'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence',
                'cvd_trend', 'oi_change', 'oi_trend', 'oi_price_divergence', 'mtf_dist_1h',
                'mtf_dist_4h', 'btc_log_return', 'btc_correlation', 'noise_index'
            ]

            input_data = pd.DataFrame([latest_row[features]], columns=features)
            if input_data.isna().any().any():
                return

            # 4. Descompacta a mente do robô
            brain = joblib.load(model_path)
            hmm = brain['hmm']
            lgbm = brain['lgbm']
            xgb = brain['xgb']
            cat = brain['catboost']
            meta = brain['meta']

            # Alimenta e calcula o regime HMM
            try:
                input_data['hmm_regime'] = hmm.predict(input_data[['log_return', 'volatility_cluster']])
            except:
                input_data['hmm_regime'] = 0

            # 5. Previsão Ensemble (O Comitê de Algoritmos)
            X_meta = np.column_stack([
                np.asarray(lgbm.predict_proba(input_data[features])),
                np.asarray(xgb.predict_proba(input_data[features])),
                np.asarray(cat.predict_proba(input_data[features]))
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

            # 6. Formatação do Sinal para o Executor
            if direction is not None:
                score = round(prob_final * 10, 1)

                if score >= Config.MIN_SCORE_ENTRY:

                    mtf_1h = latest_row.get('mtf_dist_1h', 0)
                    mtf_4h = latest_row.get('mtf_dist_4h', 0)
                    trend_1h = 'Bull' if mtf_1h > 0 else 'Bear'
                    trend_4h = 'Bull' if mtf_4h > 0 else 'Bear'

                    rsi_atual = latest_row.get('RSI_14', 50)
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
        # Configuração de timeout elástico para evitar colisão com o executor
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
        logging.info("📡 Radar Ensemble Institucional inicializado (Anti-Ban).")
        while True:
            try:
                await self.limpar_sinais_db()
                ativos = Config.get_ativos()

                for symbol in ativos:
                    await self.analisador_ativo_safe(symbol)
                    await asyncio.sleep(1.5)
            except Exception as e:
                logging.error(f"Erro no loop principal do radar: {e}")
            await asyncio.sleep(60)

    async def analisador_ativo_safe(self, symbol: str):
        try:
            await self.analisar_ativo(symbol)
        except Exception as e:
            logging.error(f"Falha segura ao processar {symbol} no radar: {e}")

# =====================================================================
# ROTINAS DE INICIALIZAÇÃO PARA CONEXÃO COM O MAIN.PY (INTACTAS)
# =====================================================================

def loop_treinador_continuo(treinador_instance):
    while True:
        try:
            treinador_instance.iniciar_ciclo_treinamento()
            time.sleep(Config.HORAS_RETREINO * 3600)
        except Exception as e:
            logging.error(f"Erro no ciclo de retreino: {e}")
            time.sleep(60)

def iniciar_motores_ia(public_exchange: ccxt.bybit):
    treinador = MotorTreinamento() # Instancia o seu treinador pesado (XGBoost, HMM)
    radar = AnalisadorRadar(public_exchange)

    # Inicia a Thread de Treinamento contínuo
    t_treino = threading.Thread(target=loop_treinador_continuo, args=(treinador,), daemon=True, name="Thread-Treinador-IA")
    t_treino.start()
    logging.info("A iniciar a thread do Treinador de IA Institucional...")

    # Função isolada para rodar o loop assíncrono do radar
    def start_radar_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(radar.loop_radar())

    # Inicia a Thread do Radar
    t_radar = threading.Thread(target=start_radar_loop, daemon=True, name="Thread-Radar-Analisador")
    t_radar.start()
    logging.info("A iniciar a thread do Analisador (Radar)...")
