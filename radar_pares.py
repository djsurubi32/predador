import os
import time
import asyncio
import logging
import joblib
import pandas as pd
import numpy as np
import aiosqlite
import ccxt.async_support as ccxt_async
from config import Config

# Simulação da importação da sua arquitetura
# from ta_indicators import adicionar_todos_indicadores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RADAR DE ARBITRAGEM] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class RadarArbitragem:
    def __init__(self):
        self.modelos_dir = Config.ARB_MODELOS_DIR
        self.db_name = Config.DB_NAME
        self.exchange = ccxt_async.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        self.modelos_ativos = {}
        self.pares_monitorados = []

    def carregar_modelos_do_dia(self):
        """
        Lê os arquivos .pkl gerados pelo Treinador de Arbitragem e 
        carrega na memória apenas os cérebros validados para o dia de hoje.
        """
        self.modelos_ativos.clear()
        self.pares_monitorados.clear()
        
        if not os.path.exists(self.modelos_dir):
            logging.warning("Diretório de modelos de arbitragem não encontrado.")
            return

        arquivos = [f for f in os.listdir(self.modelos_dir) if f.endswith('.pkl')]
        
        for arquivo in arquivos:
            try:
                # O nome do arquivo segue o padrão: arb_BTCUSDT_ETHUSDT.pkl
                partes = arquivo.replace('.pkl', '').split('_')
                if len(partes) == 3:
                    leg_long = partes[1]
                    leg_short = partes[2]
                    par_id = f"{leg_long}_{leg_short}"
                    
                    caminho_completo = os.path.join(self.modelos_dir, arquivo)
                    modelo = joblib.load(caminho_completo)
                    
                    self.modelos_ativos[par_id] = {
                        "modelo": modelo,
                        "leg_long": leg_long,
                        "leg_short": leg_short
                    }
                    self.pares_monitorados.extend([leg_long, leg_short])
                    
            except Exception as e:
                logging.error(f"Falha ao carregar o modelo {arquivo}: {e}")
                
        # Remove duplicatas da lista de moedas para otimizar os requests de API
        self.pares_monitorados = list(set(self.pares_monitorados))
        logging.info(f"✅ Modelos de pares carregados: {len(self.modelos_ativos)}. Moedas ativas no radar: {len(self.pares_monitorados)}")

    async def buscar_dados_mercado(self) -> dict:
        """
        Busca os dados mais recentes das moedas monitoradas para montar os candles.
        Em produção, recomenda-se buscar OHLCV para montar o ativo sintético completo.
        """
        dados_recentes = {}
        try:
            # Para alta performance, buscamos apenas os tickers simultâneos
            tickers = await self.exchange.fetch_tickers(self.pares_monitorados)
            for symbol, ticker in tickers.items():
                if symbol in self.pares_monitorados:
                    # Simulando um DataFrame OHLCV de 1 linha baseado no ticker atual
                    # Na prática, você puxará os últimos N candles para aplicar os 34 parâmetros
                    preco_atual = float(ticker['last'])
                    df = pd.DataFrame([{
                        'open': preco_atual,
                        'high': preco_atual * 1.001,
                        'low': preco_atual * 0.999,
                        'close': preco_atual,
                        'volume': float(ticker['quoteVolume'])
                    }])
                    dados_recentes[symbol] = df
        except Exception as e:
            logging.error(f"Erro na comunicação com a API Bybit: {e}")
            
        return dados_recentes

    def criar_candle_sintetico_tempo_real(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        """
        Mesma lógica do Treinador, mas aplicada aos dados instantâneos.
        """
        df_sintetico = pd.DataFrame()
        df_sintetico['open'] = df_a['open'] - df_b['open']
        df_sintetico['high'] = df_a['high'] - df_b['low']
        df_sintetico['low'] = df_a['low'] - df_b['high']
        df_sintetico['close'] = df_a['close'] - df_b['close']
        df_sintetico['volume'] = (df_a['volume'] + df_b['volume']) / 2
        
        return df_sintetico

    async def enviar_sinal_para_banco(self, par_id: str, leg_long: str, leg_short: str, price_long: float, price_short: float):
        """
        Comunica-se com o gerenciador.py injetando a oportunidade na tabela SQLite.
        """
        # Matemática de alvos Market Neutral. 
        # O capital de risco e alvos reais serão gerenciados dinamicamente no gerenciador.py
        # Aqui, enviamos um PnL projetado mínimo que valida a entrada.
        banca = getattr(Config, 'BANCA_DEMO_INICIAL', 100.0) 
        risco_alocado = banca * Config.ARB_MAX_POSITION_RISK
        
        target_pnl_usd = risco_alocado * 1.5  # Exemplo: Busca 1.5x o risco no Spread
        stop_pnl_usd = -risco_alocado         # Corta se o Spread dilatar contra o risco máximo

        try:
            async with aiosqlite.connect(self.db_name) as db_conn:
                await db_conn.execute(
                    '''
                    INSERT OR IGNORE INTO sinais_arbitragem 
                    (par_id, leg_long, leg_short, price_long, price_short, target_pnl_usd, stop_pnl_usd) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', 
                    (par_id, leg_long, leg_short, price_long, price_short, target_pnl_usd, stop_pnl_usd)
                )
                await db_conn.commit()
                logging.info(f"🚀 SINAL DE ARBITRAGEM GERADO: {par_id} (Long: {leg_long} @ {price_long:.4f} | Short: {leg_short} @ {price_short:.4f})")
        except Exception as e:
            logging.error(f"Erro ao inserir sinal de arbitragem no BD: {e}")

    async def iniciar_varredura(self):
        """
        Loop infinito assíncrono. O verdadeiro "Motor" do Radar.
        """
        self.carregar_modelos_do_dia()
        
        if not self.modelos_ativos:
            logging.warning("Nenhum modelo de par encontrado. O Radar entrará em espera...")
            
        while True:
            if not self.modelos_ativos:
                await asyncio.sleep(60)
                self.carregar_modelos_do_dia()
                continue

            dados_mercado = await self.buscar_dados_mercado()
            
            if not dados_mercado:
                await asyncio.sleep(Config.CICLO_SEGUNDOS)
                continue

            for par_id, info in self.modelos_ativos.items():
                leg_long = info['leg_long']
                leg_short = info['leg_short']
                modelo = info['modelo']
                
                if leg_long not in dados_mercado or leg_short not in dados_mercado:
                    continue

                df_a = dados_mercado[leg_long]
                df_b = dados_mercado[leg_short]
                
                # Monta a estrutura da diferença entre as duas moedas
                df_sintetico = self.criar_candle_sintetico_tempo_real(df_a, df_b)
                
                # Opcional: Adicionar os 34 parâmetros da sua arquitetura
                # df_sintetico = adicionar_todos_indicadores(df_sintetico)
                
                features = ['open', 'high', 'low', 'close', 'volume']
                X_real_time = df_sintetico[features]
                
                # A IA avalia se a distorção atual é um gatilho de retorno à média
                probabilidade = modelo.predict(X_real_time)[0]
                
                # Threshold de Convicção (pode ser alinhado com Config.MIN_PROB_CONVICTION)
                if probabilidade > 0.65: # 65% de certeza que o elástico vai voltar
                    preco_l = df_a['close'].iloc[-1]
                    preco_s = df_b['close'].iloc[-1]
                    
                    await self.enviar_sinal_para_banco(
                        par_id=par_id, 
                        leg_long=leg_long, 
                        leg_short=leg_short, 
                        price_long=preco_l, 
                        price_short=preco_s
                    )
            
            # Repouso térmico da VPS
            await asyncio.sleep(Config.CICLO_SEGUNDOS)

if __name__ == "__main__":
    radar = RadarArbitragem()
    try:
        asyncio.run(radar.iniciar_varredura())
    except KeyboardInterrupt:
        logging.info("Radar de Arbitragem Estatística desligado.")
