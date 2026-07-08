import os
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

def obter_melhores_moedas(limite: int = 100) -> List[str]:
    logger.info(f"🔄 Conectando à Bybit para buscar o Top {limite} de criptomoedas mais negociadas hoje...")
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

        logger.info(f"✅ Universo atualizado! O robô vai caçar nas {len(top_ativos)} moedas mais quentes do dia.")
        return top_ativos

    except Exception as e:
        logger.error(f"❌ Erro ao buscar moedas na Bybit: {e}. Acionando lista de segurança.")
        return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT']

class Config:
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    MODELS_DIR: str = "modelos_ia"

    # GESTÃO DE RISCO DO BALDE GLOBAL
    BASE_TARGET_BASKET_PROFIT: float = 5.00
    BASE_BASKET_TRAILING_PULLBACK: float = 1.50
    BASE_STOP_BASKET_LOSS: float = -10.00
    BASE_BASKET_MARGIN: float = -15.0

    # BREAKEVEN DINÂMICO
    BASE_BREAKEVEN_TRIGGER: float = 2.50
    BASE_BREAKEVEN_PROFIT: float = 1.00

    MIN_SCORE_ENTRY: float = 6.0
    MAX_PENDING_ORDER_MINUTES: int = 5

    OPERA_CONTA_REAL: bool = False
    BANCA_DEMO_INICIAL: float = 100.0
    KELLY_FRACTION: float = 0.045
    RISK_REWARD_RATIO: float = 2.0
    MAX_POSITION_RISK: float = 0.045
    MAX_OPEN_TRADES: int = 10
    ALAVANCAGEM: int = 100

    BARRIER_HORIZON: int = 20
    BARRIER_TP_PCT: float = 1.012
    BARRIER_SL_PCT: float = 0.988
    CORRELATION_THRESHOLD: float = 0.70
    MAX_TRADE_DURATION_MINUTES: int = 240

    # Calibrado para o limite estrito de 2GB de RAM do servidor
    NUM_MOEDAS_OPERACIONAIS: int = 30

    CICLO_SEGUNDOS: int = 1
    TEMPO_ESPERA_HOLD_MINUTOS: int = 5
    TIMEFRAME: str = '15m'
    CANDLES_TREINAMENTO_ML: int = 1500
    HORAS_RETREINO: int = 4
    NLP_MODEL_NAME: str = 'all-MiniLM-L6-v2'

    _ATIVOS: List[str] = []

    @classmethod
    def get_ativos(cls) -> List[str]:
        if not cls._ATIVOS:
            cls._ATIVOS = obter_melhores_moedas(cls.NUM_MOEDAS_OPERACIONAIS)
        return cls._ATIVOS

    @classmethod
    def inicializar_estrutura(cls) -> None:
        if not os.path.exists(cls.MODELS_DIR):
            os.makedirs(cls.MODELS_DIR)
            logger.info(f"📁 Diretório '{cls.MODELS_DIR}' preparado com sucesso.")

Config.inicializar_estrutura()
