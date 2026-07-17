import os
import sys
import time
import logging
import sqlite3
import asyncio
import traceback

import numpy as np
import pandas as pd
import ccxt
import joblib

from config import Config
from treinador import MotorTreinamento

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RADAR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class AnalisadorRadar:
    def __init__(self, public_exchange: ccxt.bybit, db_name: str = "predador_v31.db"):
        self.exchange = public_exchange
        self.db_name = db_name
        # Instanciamos o motor de treinamento APENAS para herdar a preparação exata de features
        # garantindo 100% de paridade matemática entre o backtest e o mercado ao vivo.
        self.motor_base = MotorTreinamento()

    async def obter_dados_recentes_async(self, symbol: str, limit: int = 150) -> pd.DataFrame:
        try:
            ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, Config.TIMEFRAME, limit=limit)

            # Busca do Open Interest assíncrono para formar as features institucionais
            try:
                oi_data = await asyncio.to_thread(self.exchange.fetch_open_interest_history, symbol, Config.TIMEFRAME, limit=limit)
                oi_map = {int(item.get('timestamp', 0)): float(item.get('openInterestValue') or item.get('info', {}).get('openInterest', 0)) for item in oi_data}
            except Exception:
                oi_map = {}

            # NOVA BUSCA: Funding Rate assíncrono para a IA ler a Caça à Liquidez
            try:
                fr_data = await asyncio.to_thread(self.exchange.fetch_funding_rate_history, symbol, limit=limit)
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

            # DataFrame atualizado com a coluna funding_rate
            df = pd.DataFrame(merged, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.ffill(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Erro ao obter dados recentes para {symbol}: {e}")
            return pd.DataFrame()

    async def analisar_ativo(self, symbol: str, btc_df: pd.DataFrame):
        safe_symbol_name = symbol.replace('/', '_').replace(':', '_')
        model_path = os.path.join(Config.MODELS_DIR, f"{safe_symbol_name}.pkl")

        if not os.path.exists(model_path):
            return

        df = await self.obter_dados_recentes_async(symbol)
        if df.empty or len(df) < 50:
            return

        # 🧠 HERANÇA INSTITUCIONAL: Aplica as 34 features (originais + 15 sensores) exatas do Treinador com Multi-Timeframe (1h/4h)
        df = self.motor_base.prepare_features(df, btc_df)

        try:
            brain = joblib.load(model_path)
            hmm_model = brain['hmm']
            lgbm = brain['lgbm']
            xgb_model = brain['xgb']
            cb_model = brain['catboost']
            meta_learner = brain['meta']
        except Exception as e:
            logging.error(f"Erro ao carregar o cérebro (Ensemble) de {symbol}: {e}")
            return

        # Identificação em tempo real do Regime de Mercado (HMM)
        try:
            df['hmm_regime'] = hmm_model.predict(df[['log_return', 'volatility_cluster']])
        except Exception:
            df['hmm_regime'] = 0

        # ARRAY ATUALIZADO: 34 Features exatas exigidas pelos modelos retreinados
        features = [
            'RSI_14', 'price_vs_ema', 'vol_zscore', 'volatility_cluster', 'bb_pos', 'ADX_14', 
            'MACD_12_26_9', 'log_return', 'hmm_regime', 'rsi_divergence', 'cvd_trend', 'oi_change', 
            'oi_trend', 'oi_price_divergence', 'mtf_dist_1h', 'mtf_dist_4h', 'btc_log_return', 
            'btc_correlation', 'noise_index',
            'obv_rsi_div', 'cvd_accel', 'vwap_dist', 'liq_vacuum', 'funding_rate_delta', 
            'oi_momentum', 'nvi', 'hurst_proxy', 'z_score_50', 'autocorr_3', 
            'fractal_dim', 'squeeze_ratio', 'natr', 'skewness_20', 'macd_hist_vel'
        ]

        latest_row = df.iloc[-1]
        if latest_row[features].isna().any():
            return

        try:
            input_data = pd.DataFrame([latest_row[features]], columns=features)

            # Extrai as probabilidades dos 3 modelos base
            prob_lgbm = lgbm.predict_proba(input_data)
            prob_xgb = xgb_model.predict_proba(input_data)
            prob_cb = cb_model.predict_proba(input_data)

            # Empilha as predições para o Meta-Learner processar
            X_meta = np.column_stack([prob_lgbm, prob_xgb, prob_cb])
            meta_probs = meta_learner.predict_proba(X_meta)[0]

            # Classes do Triple Barrier: 1 = Take Profit (BUY), 2 = Stop Loss (SELL na ótica de short)
            prob_up = float(meta_probs[1]) if len(meta_probs) > 1 else 0.0
            prob_down = float(meta_probs[2]) if len(meta_probs) > 2 else 0.0

            direction = None
            prob_final = 0.0

            # 🛡️ Trava de Confiança do Comitê de IA (> 70%)
            if prob_up >= 0.70:
                direction = 'BUY'
                prob_final = prob_up
            elif prob_down >= 0.70:
                direction = 'SELL'
                prob_final = prob_down

            if direction is not None:
                score = round(prob_final * 10, 1)

                if score >= Config.MIN_SCORE_ENTRY:

                    atr_atual = float(latest_row.get('ATRr_14', 0.0))
                    sma_atr = float(latest_row.get('SMA_ATR_100', 0.0))
                    funding_atual = float(latest_row.get('funding_rate', 0.0))

                    rsi_atual = latest_row['RSI_14']
                    fluxo_score = 3.0 if (rsi_atual > 65 or rsi_atual < 35) else 2.0 if (rsi_atual > 55 or rsi_atual < 45) else 1.0

                    bb_pos = latest_row['bb_pos']
                    espaco_score = 3.0 if (0.2 <= bb_pos <= 0.8) else 1.0

                    reasoning = (
                        f"ML_Dir:{score:.1f} | "
                        f"ML_Espaço:{espaco_score:.1f} | "
                        f"Fluxo:{fluxo_score:.1f} | "
                        f"Regime_HMM:{int(latest_row['hmm_regime'])}"
                    )

                    self.salvar_sinal_db(
                        symbol=symbol,
                        direction=direction,
                        price=float(latest_row['close']),
                        prob=prob_final * 100,
                        score=score,
                        atr=atr_atual,
                        sma_atr=sma_atr,
                        funding=funding_atual,
                        reasoning=reasoning
                    )
        except Exception as e:
            logging.error(f"Erro na matriz de predição de {symbol}: {e}")

    def salvar_sinal_db(self, symbol: str, direction: str, price: float, prob: float, score: float, atr: float, sma_atr: float, funding: float, reasoning: str):
        with sqlite3.connect(self.db_name, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO elite_signals
                (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (symbol, direction, price, prob, score, atr, sma_atr, funding, reasoning, time.time()))
            conn.commit()

    async def limpar_sinais_db(self):
        try:
            with sqlite3.connect(self.db_name, timeout=30) as conn:
                cursor = conn.cursor()
                # Limpa sinais antigos para evitar entradas atrasadas no Executor
                cursor.execute('DELETE FROM elite_signals WHERE timestamp < ?', (time.time() - 900,))
                conn.commit()
        except Exception:
            pass

    async def loop_radar(self):
        logging.info("📡 Radar Institucional inicializado com Isolamento Multiprocessado (Fila Sequencial Anti-Ban).")
        while True:
            try:
                await self.limpar_sinais_db()
                ativos = Config.get_ativos()

                if not ativos:
                    await asyncio.sleep(10)
                    continue

                # Otimização: Puxa o gráfico do BTC uma única vez no ciclo para cruzar a correlação
                btc_df = await self.obter_dados_recentes_async('BTC/USDT:USDT', limit=150)

                for symbol in ativos:
                    await self.analisador_ativo_safe(symbol, btc_df)
                    await asyncio.sleep(1.5)

            except Exception as e:
                logging.error(f"Erro no loop principal do radar: {traceback.format_exc()}")
            await asyncio.sleep(30)

    async def analisador_ativo_safe(self, symbol: str, btc_df: pd.DataFrame):
        try:
            await self.analisar_ativo(symbol, btc_df)
        except Exception as e:
            logging.error(f"Falha segura ao processar {symbol} no radar: {e}")
            
