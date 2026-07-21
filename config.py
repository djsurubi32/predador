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
    logger.info(f"🔄 Conectando à Bybit para buscar o Top {limite} de criptomoedas blindadas e líquidas...")
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

                # BLINDAGEM INSTITUCIONAL
                # 1. Rejeita moedas com Bid/Ask zerados ou falhas no book
                if bid > 0 and ask > 0:

                    # 2. Spread (Distância Bid/Ask) máximo de 0.05% (Evita Derrapagem / Slippage)
                    spread_pct = ((ask - bid) / bid) * 100

                    # 3. Volume mínimo de 55 milhões de dólares nas últimas 24h (Liquidez Extrema)
                    if spread_pct <= 0.05 and volume_24h >= 25000000:
                        moedas_validas.append((symbol, volume_24h))

        moedas_validas.sort(key=lambda x: x[1], reverse=True)
        top_ativos = [x[0] for x in moedas_validas[:limite]]

        logger.info(f"✅ Universo atualizado! {len(top_ativos)} moedas ultra-líquidas e sem risco de derrapagem detectadas.")
        return top_ativos

    except Exception as e:
        logger.error(f"❌ Erro ao buscar moedas na Bybit: {e}. Acionando lista de segurança institucional.")
        return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'BNB/USDT:USDT']

class Config:
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    MODELS_DIR: str = "modelos_ia"

    MIN_SCORE_ENTRY: float = 6.5
    OPERA_CONTA_REAL: bool = False

    # ---------------------------------------------------------
    # GESTÃO GLOBAL DO BALDE (SINCRONIZADO PARA 35X E 2 VELAS)
    # ---------------------------------------------------------
    BASKET_TARGET_PCT: float = 3.0             # Alvo: 140% ROE (Equivale aos 4% de TP da IA no preço)
    BASKET_TRAILING_PULLBACK_PCT: float = 0.15  # Trailing: Recuo de 15% após atingir o alvo
    BASKET_STOP_LOSS_PCT: float = -0.90         # Stop Global: -90% ROE (Dá 2.5% de fôlego no preço. Mantém o R:R saudável sem asfixiar)

    BASKET_BREAKEVEN_TRIGGER_PCT: float = 0.60  # Armar Catraca: Aciona apenas com 60% de lucro (Deixa a moeda explodir sem sufocar)
    BASKET_BREAKEVEN_PROFIT_PCT: float = 0.15   # Piso da Catraca: Garante 15% de lucro mínimo caso reverta

    MAX_PENDING_ORDER_MINUTES: int = 5
    BANCA_DEMO_INICIAL: float = 100.0
    KELLY_FRACTION: float = 0.045
    RISK_REWARD_RATIO: float = 2.0
    MAX_POSITION_RISK: float = 0.045

    MAX_OPEN_TRADES: int = 10
    ALAVANCAGEM: int = 35

    # ---------------------------------------------------------
    # BARREIRAS DE TREINAMENTO ML (FÔLEGO PARA 35X)
    # ---------------------------------------------------------
    BARRIER_HORIZON: int = 2
    BARRIER_TP_PCT: float = 1.040               # Alvo triplo base: 4% na moeda
    BARRIER_SL_PCT: float = 0.970               # Stop triplo base: 3% na moeda
    
    CORRELATION_THRESHOLD: float = 0.70
    MAX_TRADE_DURATION_MINUTES: int = 240

    NUM_MOEDAS_OPERACIONAIS: int = 2000
    CICLO_SEGUNDOS: int = 1
    TEMPO_ESPERA_HOLD_MINUTOS: int = 5
    TIMEFRAME: str = '15m'
    CANDLES_TREINAMENTO_ML: int = 70000
    HORAS_RETREINO: int = 10
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
