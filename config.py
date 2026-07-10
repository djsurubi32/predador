import os
import time
import logging
import ccxt
from typing import List, Tuple
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

def obter_melhores_moedas(limite: int = 30) -> List[str]:
    logger.info(f"🔄 Conectando à Bybit para buscar o Top {limite} de criptomoedas mais negociadas...")
    try:
        exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        exchange.load_markets()
        tickers = exchange.fetch_tickers()

        moedas_validas: List[Tuple[str, float]] = []
        for symbol, ticker in tickers.items():
            market = exchange.markets.get(symbol)
            if market and market.get('linear') and market.get('quote') == 'USDT' and market.get('active'):
                volume_24h = float(ticker.get('quoteVolume', 0) or 0)
                moedas_validas.append((symbol, volume_24h))

        moedas_validas.sort(key=lambda x: x[1], reverse=True)
        top_ativos = [x[0] for x in moedas_validas[:limite]]

        logger.info(f"✅ Universo rotativo atualizado! As {len(top_ativos)} moedas mais quentes agora estão no radar.")
        return top_ativos

    except Exception as e:
        logger.error(f"❌ Erro ao buscar moedas na Bybit: {e}. Acionando lista de segurança.")
        return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']

class Config:
    # CHAVES DE API E INTEGRAÇÕES
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    MODELS_DIR: str = "modelos_ia"

    # FILTRO SEVERO ATIVADO
    MIN_SCORE_ENTRY: float = 7.0
    OPERA_CONTA_REAL: bool = False

    # GESTÃO MATEMÁTICA DO BALDE DESPRESSURIZADO
    BASE_TARGET_BASKET_PROFIT: float = 4.00
    BASE_BASKET_TRAILING_PULLBACK: float = 0.80
    BASE_STOP_BASKET_LOSS: float = -5.00
    BASE_BASKET_MARGIN: float = -7.00

    BASE_BREAKEVEN_TRIGGER: float = 2.00
    BASE_BREAKEVEN_PROFIT: float = 1.00

    MAX_PENDING_ORDER_MINUTES: int = 5
    BANCA_DEMO_INICIAL: float = 100.0
    KELLY_FRACTION: float = 0.045
    RISK_REWARD_RATIO: float = 2.0
    MAX_POSITION_RISK: float = 0.045

    # LIMITE DE OPERAÇÕES SIMULTÂNEAS REDUZIDO
    MAX_OPEN_TRADES: int = 7
    ALAVANCAGEM: int = 100

    BARRIER_HORIZON: int = 20
    BARRIER_TP_PCT: float = 1.012
    BARRIER_SL_PCT: float = 0.988
    CORRELATION_THRESHOLD: float = 0.70
    MAX_TRADE_DURATION_MINUTES: int = 240

    NUM_MOEDAS_OPERACIONAIS: int = 30
    CICLO_SEGUNDOS: int = 1
    TEMPO_ESPERA_HOLD_MINUTOS: int = 5
    TIMEFRAME: str = '15m'
    CANDLES_TREINAMENTO_ML: int = 1500
    HORAS_RETREINO: int = 4
    NLP_MODEL_NAME: str = 'all-MiniLM-L6-v2'

    _ATIVOS: List[str] = []
    _ULTIMA_ATUALIZACAO: float = 0.0

    @classmethod
    def get_ativos(cls) -> List[str]:
        agora = time.time()
        if not cls._ATIVOS or (agora - cls._ULTIMA_ATUALIZACAO) > 300:
            cls._ATIVOS = obter_melhores_moedas(cls.NUM_MOEDAS_OPERACIONAIS)
            cls._ULTIMA_ATUALIZACAO = agora
        return cls._ATIVOS

    @classmethod
    def inicializar_estrutura(cls) -> None:
        if not os.path.exists(cls.MODELS_DIR):
            os.makedirs(cls.MODELS_DIR)

Config.inicializar_estrutura()
