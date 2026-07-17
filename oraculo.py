import logging
import asyncio
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ORÁCULO BINANCE] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class OraculoBinance:
    def __init__(self):
        """
        Inicializa a conexão assíncrona com o mercado de Futuros da Binance (USDT-M),
        que possui a correlação direta de preço e volume com os Swaps da Bybit.
        """
        self.exchange = ccxt_async.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        self.market_loaded = False

    async def _garantir_mercados(self):
        if not self.market_loaded:
            await self.exchange.load_markets()
            self.market_loaded = True

    async def fechar_conexoes(self):
        """Prevenção de vazamento de memória do loop assíncrono."""
        await self.exchange.close()

    def traduzir_simbolo(self, symbol_bybit: str) -> str:
        """
        Bybit usa o padrão 'BTC/USDT:USDT'.
        Binance usa o padrão 'BTC/USDT'.
        Esta função formata a string perfeitamente para o Oráculo.
        """
        return symbol_bybit.split(':')[0]

    async def analisar_imbalance_l2(self, symbol: str) -> float:
        """
        Baixa os top 50 níveis do Livro de Ofertas (Order Book Nível 2).
        Calcula o Desequilíbrio (Imbalance) financeiro entre as intenções de Compra (Bids) e Venda (Asks).
        Retorno: -1.0 (Muralha 100% vendedora) a 1.0 (Muralha 100% compradora).
        """
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=50)
            bids = np.array(ob['bids'])  # [preço, quantidade]
            asks = np.array(ob['asks'])

            vol_bids = np.sum(bids[:, 0] * bids[:, 1]) if len(bids) > 0 else 0.0
            vol_asks = np.sum(asks[:, 0] * asks[:, 1]) if len(asks) > 0 else 0.0

            # Proteção 1e-9 contra divisão por zero em micro-segundos de book vazio
            imbalance = (vol_bids - vol_asks) / (vol_bids + vol_asks + 1e-9)
            return float(imbalance)
        except Exception as e:
            logging.error(f"Falha ao ler o Livro de Ofertas L2 para {symbol}: {e}")
            return 0.0

    async def analisar_delta_volume(self, symbol: str) -> float:
        """
        Lê as últimas 15 velas de 1 minuto (15 minutos imediatos de fluxo).
        Aproxima o CVD (Cumulative Volume Delta) para medir agressão institucional recente.
        Retorno: -1.0 (Forte agressão de venda) a 1.0 (Forte agressão de compra).
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=15)
            if not ohlcv or len(ohlcv) < 5:
                return 0.0

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Cálculo de agressão vetorial
            df['dir'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
            df['vol_agressao'] = df['volume'] * df['dir']
            
            delta_total = df['vol_agressao'].sum()
            volume_total = df['volume'].sum()

            cvd_ratio = delta_total / (volume_total + 1e-9)
            return float(cvd_ratio)
        except Exception as e:
            logging.error(f"Falha ao ler o Fluxo de Volume Delta para {symbol}: {e}")
            return 0.0

    async def validar_sinal_institucional(self, symbol_bybit: str, direcao_bybit: str) -> tuple[bool, str]:
        """
        Motor Central do Oráculo. Cruza o sinal gerado na Bybit com a realidade da Binance.
        Retorna (True/False, Motivo).
        """
        await self._garantir_mercados()
        symbol_binance = self.traduzir_simbolo(symbol_bybit)

        try:
            # Executa a varredura L2 e CVD em paralelo para máxima velocidade
            tarefa_l2 = self.analisar_imbalance_l2(symbol_binance)
            tarefa_cvd = self.analisar_delta_volume(symbol_binance)
            
            imbalance_l2, cvd_ratio = await asyncio.gather(tarefa_l2, tarefa_cvd)

            aprovado = False
            motivo = ""

            # 🛡️ AJUSTE INSTITUCIONAL: Travas ultra-rígidas contra Squeeze Direcional
            if direcao_bybit == 'BUY':
                # Exige que não haja muralhas de venda fortes nem fluxo vendedor engolindo a compra (-0.05)
                if imbalance_l2 >= -0.05 and cvd_ratio >= -0.05:
                    aprovado = True
                    motivo = f"Validado Binance (L2 Imbalance: {imbalance_l2:.2f} | CVD 1m: {cvd_ratio:.2f})"
                else:
                    motivo = f"Bloqueado Binance (Muralha Vendedora {imbalance_l2:.2f} ou Despejo no Volume {cvd_ratio:.2f})"
            
            elif direcao_bybit == 'SELL':
                # Exige que não haja muralhas de compra fortes nem fluxo comprador forte engolindo a venda (0.05)
                if imbalance_l2 <= 0.05 and cvd_ratio <= 0.05:
                    aprovado = True
                    motivo = f"Validado Binance (L2 Imbalance: {imbalance_l2:.2f} | CVD 1m: {cvd_ratio:.2f})"
                else:
                    motivo = f"Bloqueado Binance (Muralha Compradora {imbalance_l2:.2f} ou Absorção no Volume {cvd_ratio:.2f})"

            return aprovado, motivo

        except Exception as e:
            logging.error(f"Erro Crítico no cruzamento do Oráculo para {symbol_binance}: {e}")
            # Em caso de falha de API do Oráculo, liberamos a trade para não paralisar o Predador primário
            return True, "Oráculo Indisponível - Bypass Automático"

# --- BLOCO DE TESTE INDEPENDENTE ---
if __name__ == "__main__":
    async def teste_oraculo():
        oraculo = OraculoBinance()
        
        aprovado, razao = await oraculo.validar_sinal_institucional('BTC/USDT:USDT', 'BUY')
        print(f"Teste BUY BTC: Aprovado={aprovado} | Razão: {razao}")
        
        aprovado, razao = await oraculo.validar_sinal_institucional('ETH/USDT:USDT', 'SELL')
        print(f"Teste SELL ETH: Aprovado={aprovado} | Razão: {razao}")

        await oraculo.fechar_conexoes()

    asyncio.run(teste_oraculo())
