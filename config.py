import os
import time
import logging
import asyncio
import ccxt.async_support as ccxt_async
from typing import List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

class BotConfig(BaseSettings):
    # Credenciais
    BYBIT_API_KEY: str = Field(default="")
    BYBIT_SECRET: str = Field(default="")
    TELEGRAM_TOKEN: str = Field(default="")
    TELEGRAM_CHAT_ID: str = Field(default="")

    # Infraestrutura
    DB_NAME: str = Field(default="predador_v31.db")
    MODELS_DIR: str = Field(default="modelos_ia")
    MAX_MODELS_IN_RAM: int = Field(default=30)
    NLP_MODEL_NAME: str = Field(default="ProsusAI/finbert")

    OPERA_CONTA_REAL: bool = Field(default=False)
    BANCA_DEMO_INICIAL: float = Field(default=100.0)

    # ---------------------------------------------------------
    # MÓDULO DE ARBITRAGEM ESTATÍSTICA (PAIRS TRADING & IA)
    # ---------------------------------------------------------
    ARB_MODELOS_DIR: str = Field(default="modelos_arbitragem")
    ARB_DIAS_HISTORICO: int = Field(default=30)                  # Janela de 30 dias para capturar regimes atuais
    ARB_P_VALUE_THRESHOLD: float = Field(default=0.05)           # Nível de 95% de confiança na Cointegração do par
    ARB_MAX_PARES_ATIVOS: int = Field(default=5)                 # Limite de pares (10 ordens simultâneas no total)
    
    # Parâmetros do Motor de Clustering (Fase 1)
    ARB_DBSCAN_EPS: float = Field(default=0.5)                   # Distância máxima de correlação no espaço vetorial
    ARB_DBSCAN_MIN_SAMPLES: int = Field(default=2)               # Mínimo de 2 moedas para consolidar um grupo
    
    # Gerenciamento de Risco Isolado da Arbitragem
    ARB_MAX_PORTFOLIO_RISK: float = Field(default=0.04, le=0.15) # Risco máximo global dedicado apenas à arbitragem
    ARB_MAX_POSITION_RISK: float = Field(default=0.01, le=0.05)  # Risco máximo por par (engloba as pernas Long e Short)

    # ---------------------------------------------------------
    # KILL SWITCH & SOBREVIVÊNCIA (Circuit Breakers)
    # ---------------------------------------------------------
    MAX_DAILY_DRAWDOWN: float = Field(default=0.05, le=0.10)     # Para operações se equity cair 5% no dia
    MAX_CONSECUTIVE_LOSSES: int = Field(default=5)               # Pausa sistema após 5 perdas seguidas
    MIN_LIQUIDITY_SOURCES: int = Field(default=2)                # Exige consenso de liquidez institucional
    STALE_DATA_MAX_SEC: int = Field(default=30)                  # Recusa operar com livro de ofertas defasado
    TAKER_FEE_PCT: float = Field(default=0.055)                  # Taxa real por perna (Ex: Bybit)
    SLIPPAGE_BPS_ESTIMATE: int = Field(default=3)                # Derrapagem estimada de 3 basis points (0.03%)

    # ---------------------------------------------------------
    # GERENCIAMENTO DE RISCO E DIMENSIONAMENTO (PORTFÓLIO DIRECIONAL)
    # ---------------------------------------------------------
    MAX_PORTFOLIO_RISK: float = Field(default=0.06, le=0.20)
    MAX_POSITION_RISK: float = Field(default=0.01, le=0.05)
    MAX_LEVERAGE_TOTAL: float = Field(default=3.0)
    MAX_CORRELATED_EXPOSURE: float = Field(default=0.03)
    MAX_OPEN_TRADES: int = Field(default=5)

    # Barreiras do Analisador (Corte Dinâmico)
    MIN_SCORE_ENTRY: float = Field(default=5.0)
    MIN_PROB_PREDICT: float = Field(default=55.0)                # Ajustado para realidade CalibratedClassifierCV
    MIN_PROB_CONVICTION: float = Field(default=55.0)

    # Tiers realistas para modelos empiricamente calibrados
    PROB_TIER_4: float = Field(default=70.0)
    PROB_TIER_3: float = Field(default=65.0)
    PROB_TIER_2: float = Field(default=55.0)

    MIN_EXPECTED_MOVE_PCT: float = Field(default=0.4)
    ALVO_TIER_3: float = Field(default=1.5)
    ALVO_TIER_2: float = Field(default=0.8)

    BB_POS_TOP_VETO: float = Field(default=0.95)
    BB_POS_BOTTOM_VETO: float = Field(default=0.05)

    # ---------------------------------------------------------
    # BARREIRAS DE TREINAMENTO E MACHINE LEARNING
    # ---------------------------------------------------------
    BARRIER_HORIZON: int = Field(default=36)                     # Horizonte de 1 hora no 5m
    TP_ATR_MULT: float = Field(default=2.0)
    SL_ATR_MULT: float = Field(default=1.0)

    # Correção do Data Leakage e Horizonte
    CANDLES_TREINAMENTO_ML: int = Field(default=50000)           # ~6 meses de histórico para cobrir todos os regimes
    PURGED_CV_EMBARGO_PCT: float = Field(default=0.01)           # Margem de expurgo para k-fold sem autocorrelação

    # Estrutura de Dois Níveis (Rate Limit Tracker)
    NUM_MOEDAS_RADAR: int = Field(default=500)
    NUM_MOEDAS_OPERACIONAIS: int = Field(default=30)             # Alinhado com a capacidade de RAM/Modelos

    CICLO_SEGUNDOS: int = Field(default=5)
    TEMPO_ESPERA_HOLD_MINUTOS: int = Field(default=5)
    TIMEFRAME: str = Field(default='5m')
    HORAS_RETREINO: int = Field(default=24)

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @model_validator(mode='after')
    def validate_operational_safety(self):
        if self.OPERA_CONTA_REAL and not (self.BYBIT_API_KEY and self.BYBIT_SECRET):
            raise ValueError("CRÍTICO: OPERA_CONTA_REAL=True ativado, mas as chaves de API estão vazias.")

        if (self.MAX_OPEN_TRADES * self.MAX_POSITION_RISK) > self.MAX_PORTFOLIO_RISK:
            raise ValueError("CRÍTICO: O risco máximo por trade (vezes limite posições) estoura o teto do portfólio direcional.")
            
        # NOVA VALIDAÇÃO PARA ARBITRAGEM
        if (self.ARB_MAX_PARES_ATIVOS * self.ARB_MAX_POSITION_RISK) > self.ARB_MAX_PORTFOLIO_RISK:
            raise ValueError("CRÍTICO: O risco da arbitragem alocado supera o limite permitido (ARB_MAX_PORTFOLIO_RISK).")

        custo_total_estimado_pct = (self.TAKER_FEE_PCT * 2) + (self.SLIPPAGE_BPS_ESTIMATE / 100)
        if self.MIN_EXPECTED_MOVE_PCT <= (custo_total_estimado_pct * 1.5):
            raise ValueError(f"CRÍTICO: Movimento mínimo esperado ({self.MIN_EXPECTED_MOVE_PCT}%) não cobre as taxas e derrapagem ({custo_total_estimado_pct}%).")

        return self

Config = BotConfig()

# ---------------------------------------------------------
# CRIAÇÃO AUTOMÁTICA DE DIRETÓRIOS
# ---------------------------------------------------------
if not os.path.exists(Config.MODELS_DIR):
    os.makedirs(Config.MODELS_DIR)

if not os.path.exists(Config.ARB_MODELOS_DIR):
    os.makedirs(Config.ARB_MODELOS_DIR)

class UniverseProvider:
    def __init__(self, ttl: int = 300):
        self._symbols: List[str] = []
        self._lock = asyncio.Lock()
        self._ts: float = 0.0
        self._ttl = ttl

    async def _fetch_markets(self) -> List[str]:
        logger.info(f"🔄 Escaneando todo o mercado para filtrar as {Config.NUM_MOEDAS_OPERACIONAIS} moedas mais líquidas...")
        exchange = ccxt_async.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

        try:
            await exchange.load_markets()
            tickers = await exchange.fetch_tickers()

            moedas_validas: List[Tuple[str, float]] = []
            for symbol, ticker in tickers.items():
                market = exchange.markets.get(symbol)
                if market and market.get('linear') and market.get('quote') == 'USDT' and market.get('active'):
                    volume_24h = float(ticker.get('quoteVolume', 0) or 0)
                    bid = float(ticker.get('bid', 0) or 0)
                    ask = float(ticker.get('ask', 0) or 0)

                    if bid > 0 and ask > 0:
                        spread_pct = ((ask - bid) / bid) * 100
                        if spread_pct <= 0.05 and volume_24h >= 55000000:
                            moedas_validas.append((symbol, volume_24h))

            moedas_validas.sort(key=lambda x: x[1], reverse=True)
            top_ativos = [x[0] for x in moedas_validas[:Config.NUM_MOEDAS_OPERACIONAIS]]

            if not top_ativos:
                logger.warning("Filtro de liquidez resultou vazio. Mercado colapsou ou dados de API inválidos.")
                return []

            return top_ativos

        except Exception as e:
            logger.critical(f"❌ Falha de comunicação com a Exchange ao buscar tickers: {e}")
            return []
        finally:
            await exchange.close()

    async def get_ativos(self) -> List[str]:
        async with self._lock:
            agora = time.monotonic()

            if not self._symbols or (agora - self._ts) > self._ttl:
                novos_ativos = await self._fetch_markets()

                if novos_ativos:
                    self._symbols = novos_ativos
                    self._ts = agora
                    logger.info(f"✅ Universo validado! Operando focado nas {len(self._symbols)} moedas primárias.")
                else:
                    if self._symbols:
                        logger.error("⚠️ Falha na obtenção do universo. Mantendo estado anterior (stale data) por segurança.")
                    else:
                        logger.critical("💀 Boot Crítico: Impossível construir universo de operação válido. Trave o bot.")

            return self._symbols

universe_provider = UniverseProvider()
