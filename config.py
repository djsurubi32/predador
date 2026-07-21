import os
import time
import logging
import ccxt
from typing import List, Tuple
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

load_dotenv()

def obter_melhores_moedas(limite: int = 30) -> List[str]:
    logger.info(f"🔄 Conectando à Bybit para buscar o Top {limite} de criptomoedas...")
    try:
        exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        exchange.load_markets()
        tickers = exchange.fetch_tickers()

        moedas_validas: List[Tuple[str, float]] = []
        for symbol, ticker in tickers.items():
            market = exchange.markets.get(symbol)
            if market and market.get('linear') and market.get('quote') == 'USDT' and market.get('active'):
                volume_24h = float(ticker.get('quoteVolume', 0) or 0)
                bid = float(ticker.get('bid', 0) or 0)
                ask = float(ticker.get('ask', 0) or 0)
                if bid > 0 and ask > 0:
                    spread_pct = ((ask - bid) / bid) * 100
                    if spread_pct <= 0.05 and volume_24h >= 25000000:
                        moedas_validas.append((symbol, volume_24h))

        moedas_validas.sort(key=lambda x: x[1], reverse=True)
        top_ativos = [x[0] for x in moedas_validas[:limite]]
        logger.info(f"✅ {len(top_ativos)} moedas ultra-líquidas selecionadas.")
        return top_ativos
    except Exception as e:
        logger.error(f"Erro ao buscar moedas: {e}")
        return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'BNB/USDT:USDT']

class Config:
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    MODELS_DIR: str = "modelos_ia"
    MIN_SCORE_ENTRY: float = 6.5
    OPERA_CONTA_REAL: bool = False

    # Gestão de Risco Basket (ajustável)
    BASKET_TARGET_PCT: float = 3.0
    BASKET_TRAILING_PULLBACK_PCT: float = 0.15
    BASKET_STOP_LOSS_PCT: float = -0.90
    BASKET_BREAKEVEN_TRIGGER_PCT: float = 0.60
    BASKET_BREAKEVEN_PROFIT_PCT: float = 0.15

    MAX_OPEN_TRADES: int = 10
    ALAVANCAGEM: int = 35
    KELLY_FRACTION: float = 0.045
    MAX_POSITION_RISK: float = 0.045

    BARRIER_HORIZON: int = 2
    BARRIER_TP_PCT: float = 1.040
    BARRIER_SL_PCT: float = 0.970

    TIMEFRAME: str = '15m'
    HORAS_RETREINO: int = 10
    CICLO_SEGUNDOS: int = 1
    NUM_MOEDAS_OPERACIONAIS: int = 2000
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
