import os
import time
import logging
import asyncio
import ccxt.async_support as ccxt_async
from typing import List, Tuple
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


class BotConfig(BaseSettings):
    # =====================================================================
    # CREDENCIAIS
    # =====================================================================
    BYBIT_API_KEY: str = Field(default="")
    BYBIT_SECRET: str = Field(default="")
    TELEGRAM_TOKEN: str = Field(default="")
    TELEGRAM_CHAT_ID: str = Field(default="")

    # =====================================================================
    # INFRAESTRUTURA
    # =====================================================================
    DB_NAME: str = Field(default="predador_v32.db")
    MODELS_DIR: str = Field(default=str(BASE_DIR / "modelos_ia"))
    MAX_MODELS_IN_RAM: int = Field(default=30)

    OPERA_CONTA_REAL: bool = Field(default=False)  # False = testnet Bybit
    BANCA_INICIAL_USD: float = Field(default=100.0)

    # =====================================================================
    # UNIVERSO — MAJORS + ALTS LIQUIDAS (memecoin entra se tiver volume)
    # =====================================================================
    UNIVERSE_MIN_VOLUME_24H: float = Field(default=50_000_000)  # 50M USDT
    UNIVERSE_MAX_SPREAD_PCT: float = Field(default=0.05)
    NUM_MOEDAS_OPERACIONAIS: int = Field(default=20)

    # =====================================================================
    # CICLO E TIMEFRAME — SCALP MEDIO 5m
    # =====================================================================
    TIMEFRAME: str = Field(default="5m")
    CICLO_MS: int = Field(default=2000, ge=500, le=60_000)

    # =====================================================================
    # BARREIRA DE PRIMEIRO TOQUE (O RÓTULO DO MODELO)
    # Primeiro toque em +X% ou -X% a partir do preco da vela i.
    # Simetrica e percentual: identica para BTC, memecoin e altcoin.
    # =====================================================================
    BARRIER_PCT: float = Field(default=0.30)          # +/-0.30%
    LABEL_HORIZON_CANDLES: int = Field(default=2)     # proximas 2 velas (10 min)
    TIME_STOP_MINUTES: int = Field(default=15)        # 3 velas de 5m no maximo

    # =====================================================================
    # GATE DE ENTRADA — SO ENTRA COM PROBABILIDADE CALIBRADA ALTA
    # =====================================================================
    MIN_PROB_ENTRY: float = Field(default=65.0)       # 0-100, probabilidade calibrada
    PROB_MEIA_POSICAO: float = Field(default=60.0)    # reservado (fase 2)
    USA_MEIA_POSICAO: bool = Field(default=False)

    # =====================================================================
    # RISCO — BANCA $100, 1 POSICAO, RISCO 1% = $1 POR TRADE
    # =====================================================================
    MAX_OPEN_TRADES: int = Field(default=1)
    MAX_POSITION_RISK: float = Field(default=0.01, le=0.03)   # 1% = $1
    MAX_LEVERAGE_TOTAL: float = Field(default=5.0)            # sizing implicito: 1/0.30%

    # =====================================================================
    # KILL SWITCH
    # =====================================================================
    MAX_DAILY_DRAWDOWN: float = Field(default=0.06, le=0.12)  # $6 no dia
    MAX_CONSECUTIVE_LOSSES: int = Field(default=5)
    COOLDOWN_POS_LOSS_MIN: int = Field(default=15)   # pausa no simbolo apos stop

    # =====================================================================
    # CUSTOS — MAKER-FIRST (ordem limit postOnly na entrada)
    # =====================================================================
    MAKER_FEE_PCT: float = Field(default=0.02)
    TAKER_FEE_PCT: float = Field(default=0.055)
    SLIPPAGE_MAKER_PCT: float = Field(default=0.01)   # baixo: ordem no book
    SLIPPAGE_TAKER_PCT: float = Field(default=0.03)

    ENTRY_MAKER_TIMEOUT_SEC: float = Field(default=20.0)  # fallback p/ taker
    EXIT_MAKER_TIMEOUT_SEC: float = Field(default=15.0)   # fechamento maker-first

    # =====================================================================
    # FILTROS DO ANALISADOR (alinhados ao payoff 1:1)
    # Custos maker = 0.03% ida+volta. Barreira 0.30%.
    # =====================================================================
    MIN_EXPECTED_MOVE_PCT: float = Field(default=0.30)  # >= barreira
    BB_POS_TOP_VETO: float = Field(default=0.95)   # nao compra no topo da BB
    BB_POS_BOTTOM_VETO: float = Field(default=0.05)

    # =====================================================================
    # ORACULO (L2 + CVD Binance) — tolerancias p/ 5m
    # =====================================================================
    ORACULO_L2_TOL: float = Field(default=0.02)   # nao pode estar contra
    ORACULO_CVD_TOL: float = Field(default=0.03)

    # =====================================================================
    # TREINO
    # =====================================================================
    CANDLES_TREINAMENTO: int = Field(default=100_000)  # ~347 dias de 5m
    HORAS_RETREINO: int = Field(default=12)
    PURGED_CV_EMBARGO_PCT: float = Field(default=0.01)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_operational_safety(self):
        if self.OPERA_CONTA_REAL and not (self.BYBIT_API_KEY and self.BYBIT_SECRET):
            raise ValueError(
                "CRITICO: OPERA_CONTA_REAL=True sem chaves Bybit no .env."
            )

        if self.MAX_OPEN_TRADES != 1:
            raise ValueError("CRITICO: este release opera com 1 posicao por vez.")

        # ---- Barreira vs custos: breakeven de acerto nao pode inviabilizar o gate ----
        custo_maker_rt = (self.MAKER_FEE_PCT + self.SLIPPAGE_MAKER_PCT) * 2
        breakeven_win = (self.BARRIER_PCT + custo_maker_rt) / (2 * self.BARRIER_PCT)
        if self.MIN_PROB_ENTRY / 100.0 <= breakeven_win + 0.02:
            raise ValueError(
                f"CRITICO: gate {self.MIN_PROB_ENTRY}% esta a menos de 2 pontos do "
                f"breakeven ({breakeven_win:.1%}). Suba BARRIER_PCT ou abaixe custos."
            )

        if self.TIME_STOP_MINUTES < self.LABEL_HORIZON_CANDLES * 5:
            raise ValueError(
                "CRITICO: time-stop menor que o horizonte do rotulo. "
                "O modelo nao tera tempo de realizar a barreira."
            )

        if self.BARRIER_PCT < 4 * custo_maker_rt:
            raise ValueError(
                f"CRITICO: barreira {self.BARRIER_PCT}% menor que 4x os custos "
                f"({custo_maker_rt * 4:.3f}%). Custos comem o edge."
            )

        if not (50.0 <= self.MIN_PROB_ENTRY <= 95.0):
            raise ValueError("CRITICO: MIN_PROB_ENTRY deve estar entre 50 e 95.")

        # Sizing: risco/barreira nao pode estourar a alavancagem maxima
        alavancagem_implicita = self.MAX_POSITION_RISK / (self.BARRIER_PCT / 100.0)
        if alavancagem_implicita > self.MAX_LEVERAGE_TOTAL:
            raise ValueError(
                f"CRITICO: sizing implicito exige {alavancagem_implicita:.1f}x "
                f"mas o teto e {self.MAX_LEVERAGE_TOTAL}x."
            )

        return self


Config = BotConfig()

os.makedirs(Config.MODELS_DIR, exist_ok=True)


class UniverseProvider:
    """Universo unico para treino, analise e execucao com fallback anti-queda."""

    # Ativos de seguranca caso ocorra falha de rede/DNS momentanea com a API
    FALLBACK_MAJORS = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
        "BNB/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    ]

    def __init__(self, ttl: int = 300):
        self._symbols: List[str] = []
        self._lock = asyncio.Lock()
        self._ts: float = 0.0
        self._ttl = ttl

    async def _fetch_markets(self) -> List[str]:
        logger.info(
            f"[UNIVERSO] Escaneando Bybit — top {Config.NUM_MOEDAS_OPERACIONAIS} "
            f"(vol >= {Config.UNIVERSE_MIN_VOLUME_24H/1e6:.0f}M USDT)..."
        )

        # Carrega estritamente futuros lineares (sem opções nem spot para evitar erros)
        exchange = ccxt_async.bybit({
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "fetchMarkets": ["linear"],
            },
        })

        try:
            try:
                await exchange.load_markets()
            except Exception:
                # Fallback para o servidor espelho oficial da Bybit se api.bybit.com falhar
                exchange.hostname = "bytick.com"
                await exchange.load_markets()

            tickers = await exchange.fetch_tickers()

            validas: List[Tuple[str, float]] = []
            for symbol, ticker in tickers.items():
                market = exchange.markets.get(symbol)
                if not market:
                    continue
                if not (market.get("linear") and market.get("quote") == "USDT"
                        and market.get("active")):
                    continue

                volume_24h = float(ticker.get("quoteVolume", 0) or 0)
                bid = float(ticker.get("bid", 0) or 0)
                ask = float(ticker.get("ask", 0) or 0)
                if bid <= 0 or ask <= 0:
                    continue

                spread_pct = ((ask - bid) / bid) * 100.0
                if (spread_pct <= Config.UNIVERSE_MAX_SPREAD_PCT
                        and volume_24h >= Config.UNIVERSE_MIN_VOLUME_24H):
                    validas.append((symbol, volume_24h))

            validas.sort(key=lambda x: x[1], reverse=True)
            top = [s for s, _ in validas[: Config.NUM_MOEDAS_OPERACIONAIS]]
            if not top:
                logger.warning("[UNIVERSO] Filtro de liquidez vazio, aplicando fallback seguro.")
                return self.FALLBACK_MAJORS[:Config.NUM_MOEDAS_OPERACIONAIS]
            return top
        except Exception as e:
            logger.warning(f"[UNIVERSO] Falha ao escanear Bybit ({e}). Usando lista de majors.")
            return self.FALLBACK_MAJORS[:Config.NUM_MOEDAS_OPERACIONAIS]
        finally:
            await exchange.close()

    async def get_ativos(self) -> List[str]:
        async with self._lock:
            agora = time.monotonic()
            if not self._symbols or (agora - self._ts) > self._ttl:
                novos = await self._fetch_markets()
                if novos:
                    self._symbols = novos
                    self._ts = agora
                    logger.info(
                        f"[UNIVERSO] {len(self._symbols)} ativos carregados | tf={Config.TIMEFRAME} "
                        f"| barreira +/-{Config.BARRIER_PCT}%"
                    )
                else:
                    self._symbols = self.FALLBACK_MAJORS[:Config.NUM_MOEDAS_OPERACIONAIS]
            return self._symbols


universe_provider = UniverseProvider()
