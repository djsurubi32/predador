import os
import sys
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
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ANALISADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class IndicadoresMao:
    @staticmethod
    def calcular_15m(df: pd.DataFrame) -> pd.DataFrame:
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        df['sma_atr'] = df['atr'].rolling(100).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # VWAP Móvel (Rolling 20)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
        df['vwap_dev'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)

        return df

    @staticmethod
    def calcular_macro(df: pd.DataFrame, sufixo: str) -> pd.DataFrame:
        # Tendência por Média Móvel Simples (SMA 20)
        df[f'sma_20_{sufixo}'] = df['close'].rolling(20).mean()
        df[f'sma_trend_{sufixo}'] = (df['close'] > df[f'sma_20_{sufixo}']).astype(int)

        # RSI Macro
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df[f'rsi_{sufixo}'] = 100 - (100 / (1 + rs))

        return df

class TreinadorIA:
    def __init__(self, exchange: ccxt.bybit, db_name: str = "predador_v31.db"):
        self.exchange = exchange
        self.db_name = db_name

    def obter_historico(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = df['timestamp'].astype(float)
            return df
        except Exception as e:
            logging.error(f"Erro ao obter histórico de {symbol} ({timeframe}): {e}")
            return pd.DataFrame()

    def preparar_dataset(self, symbol: str) -> pd.DataFrame:
        df_15m = self.obter_historico(symbol, '15m', Config.CANDLES_TREINAMENTO_ML)
        df_1h = self.obter_historico(symbol, '1h', int(Config.CANDLES_TREINAMENTO_ML / 4))
        df_4h = self.obter_historico(symbol, '4h', int(Config.CANDLES_TREINAMENTO_ML / 16))

        if df_15m.empty or df_1h.empty or df_4h.empty:
            return pd.DataFrame()

        df_15m = IndicadoresMao.calcular_15m(df_15m)
        df_1h = IndicadoresMao.calcular_macro(df_1h, '1h')
        df_4h = IndicadoresMao.calcular_macro(df_4h, '4h')

        df_15m = df_15m.sort_values('timestamp')
        df_1h = df_1h.sort_values('timestamp')
        df_4h = df_4h.sort_values('timestamp')

        cols_1h = ['timestamp', 'rsi_1h', 'sma_trend_1h']
        cols_4h = ['timestamp', 'rsi_4h', 'sma_trend_4h']

        df_merged = pd.merge_asof(df_15m, df_1h[cols_1h], on='timestamp', direction='backward')
        df_merged = pd.merge_asof(df_merged, df_4h[cols_4h], on='timestamp', direction='backward')

        df_merged['target'] = (df_merged['close'].shift(-4) > df_merged['close']).astype(int)
        return df_merged

    def treinar_modelo_ativo(self, symbol: str):
        df = self.preparar_dataset(symbol)
        if df.empty or len(df) < 200:
            logging.warning(f"Dados insuficientes para treinar {symbol}.")
            return

        features = ['rsi', 'atr', 'vwap_dev', 'rsi_1h', 'sma_trend_1h', 'rsi_4h', 'sma_trend_4h']
        df_clean = df.dropna(subset=features + ['target'])

        X = df_clean[features]
        y = df_clean['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

        model = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

        try:
            model.fit(X_train, y_train)
            os.makedirs(Config.MODELS_DIR, exist_ok=True)
            model_name = symbol.replace('/', '_').replace(':', '_')
            model_path = os.path.join(Config.MODELS_DIR, f"model_{model_name}.pkl")

            joblib.dump(model, model_path)
            logging.info(f"✅ Cérebro de {symbol} treinado e salvo com sucesso.")
        except Exception as e:
            logging.error(f"Erro no treinamento de {symbol}: {e}")

    def treinar_todos(self):
        ativos = Config.get_ativos()
        logging.info(f"🚀 Iniciando Treinador Quantitativo 10/10 (Lote de {len(ativos)} moedas)...")
        for symbol in ativos:
            self.treinar_modelo_ativo(symbol)
            time.sleep(0.1)
        logging.info(f"💤 Treinamento concluído. O motor vai hibernar por {Config.HORAS_RETREINO} horas.")

    def loop_treinador(self):
        while True:
            try:
                self.treinar_todos()
            except Exception as e:
                logging.error(f"Erro no ciclo de retreino: {e}")
            time.sleep(Config.HORAS_RETREINO * 3600)


class AnalisadorRadar:
    def __init__(self, exchange: ccxt.bybit, db_name: str = "predador_v31.db"):
        self.exchange = exchange
        self.db_name = db_name

    async def obter_dados_recentes_async(self, symbol: str) -> pd.DataFrame:
        try:
            ohlcv_15m = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '15m', limit=50)
            ohlcv_1h = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '1h', limit=50)
            ohlcv_4h = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, '4h', limit=50)

            df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df_15m = IndicadoresMao.calcular_15m(df_15m)
            df_1h = IndicadoresMao.calcular_macro(df_1h, '1h')
            df_4h = IndicadoresMao.calcular_macro(df_4h, '4h')

            df_15m = df_15m.sort_values('timestamp')
            df_1h = df_1h.sort_values('timestamp')
            df_4h = df_4h.sort_values('timestamp')

            df_merged = pd.merge_asof(df_15m, df_1h[['timestamp', 'rsi_1h', 'sma_trend_1h']], on='timestamp', direction='backward')
            df_merged = pd.merge_asof(df_merged, df_4h[['timestamp', 'rsi_4h', 'sma_trend_4h']], on='timestamp', direction='backward')

            return df_merged
        except Exception as e:
            logging.error(f"Erro ao obter dados recentes para {symbol}: {e}")
            return pd.DataFrame()

    async def analisar_ativo(self, symbol: str):
        model_name = symbol.replace('/', '_').replace(':', '_')
        model_path = os.path.join(Config.MODELS_DIR, f"model_{model_name}.pkl")

        if not os.path.exists(model_path):
            return

        df = await self.obter_dados_recentes_async(symbol)
        if df.empty or len(df) < 2:
            return

        latest_row = df.iloc[-1]

        features = ['rsi', 'atr', 'vwap_dev', 'rsi_1h', 'sma_trend_1h', 'rsi_4h', 'sma_trend_4h']
        if latest_row[features].isna().any():
            return

        try:
            model = joblib.load(model_path)
            input_data = pd.DataFrame([latest_row[features]], columns=features)

            prob_up = float(model.predict_proba(input_data)[0][1])

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

                    # Cálculo Dinâmico do Espaço VWAP
                    pnl_dev_vwap = latest_row['vwap_dev'] * 100
                    abs_dev = abs(pnl_dev_vwap)
                    if abs_dev < 0.5:
                        espaco_score = 3.0
                    elif abs_dev < 1.5:
                        espaco_score = 2.0
                    else:
                        espaco_score = 1.0

                    # Cálculo Dinâmico de Fluxo Institucional (Força RSI)
                    rsi_atual = latest_row['rsi']
                    if rsi_atual > 65 or rsi_atual < 35:
                        fluxo_score = 3.0
                    elif rsi_atual > 55 or rsi_atual < 45:
                        fluxo_score = 2.0
                    else:
                        fluxo_score = 1.0

                    reasoning = (
                        f"ML_Dir:{score:.1f} | "
                        f"ML_Espaço:{espaco_score:.1f}({pnl_dev_vwap:.1f}%) | "
                        f"Fluxo:{fluxo_score:.1f} | "
                        f"Macro1h:{'Bull' if latest_row['sma_trend_1h'] == 1 else 'Bear'} | "
                        f"Macro4h:{'Bull' if latest_row['sma_trend_4h'] == 1 else 'Bear'}"
                    )

                    self.salvar_sinal_db(
                        symbol=symbol,
                        direction=direction,
                        price=float(latest_row['close']),
                        prob=prob_final * 100,
                        score=score,
                        atr=float(latest_row['atr']),
                        sma_atr=float(latest_row['sma_atr']),
                        reasoning=reasoning
                    )
        except Exception as e:
            logging.error(f"Erro ao processar predição de {symbol}: {e}")

    def salvar_sinal_db(self, symbol: str, direction: str, price: float, prob: float, score: float, atr: float, sma_atr: float, reasoning: str):
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
        except Exception:
            pass

    async def loop_radar(self):
        logging.info("📡 Radar inicializado com Isolamento Multiprocessado (Fila Sequencial Anti-Ban).")
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

def iniciar_motores_ia(public_exchange: ccxt.bybit):
    treinador = TreinadorIA(public_exchange)
    radar = AnalisadorRadar(public_exchange)

    t_treino = threading.Thread(target=treinador.loop_treinador, daemon=True, name="Thread-Treinador-IA")
    t_treino.start()
    logging.info("A iniciar a thread do Treinador de IA...")

    def start_radar_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(radar.loop_radar())

    t_radar = threading.Thread(target=start_radar_loop, daemon=True, name="Thread-Radar-Analisador")
    t_radar.start()
    logging.info("A iniciar a thread do Analisador (Radar)...")
