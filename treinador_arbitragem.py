import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
import asyncio
import os
import joblib

# Assumindo as importações da sua arquitetura atual
# from ta_indicators import adicionar_todos_indicadores
# from universe_provider import UniverseProvider

class TreinadorArbitragem:
    def __init__(self, dias_historico=30, p_value_threshold=0.05):
        self.dias_historico = dias_historico
        self.p_value_threshold = p_value_threshold
        self.modelos_dir = "modelos_arbitragem"
        os.makedirs(self.modelos_dir, exist_ok=True)

    async def obter_dados_historicos(self):
        """
        Simula a busca de dados de múltiplos ativos para os últimos 30 dias.
        Na prática, chamaria seu UniverseProvider.
        """
        print(f"Baixando histórico de {self.dias_historico} dias para análise de pares...")
        # Simulação de dados para exemplo
        tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'LINKUSDT', 'UNIUSDT']
        datas = pd.date_range(end=pd.Timestamp.now(), periods=self.dias_historico * 24, freq='H')
        
        dados = {}
        for ticker in tickers:
            df = pd.DataFrame({
                'open': np.random.uniform(100, 200, len(datas)),
                'high': np.random.uniform(100, 200, len(datas)) + 5,
                'low': np.random.uniform(100, 200, len(datas)) - 5,
                'close': np.random.uniform(100, 200, len(datas)),
                'volume': np.random.uniform(1000, 5000, len(datas))
            }, index=datas)
            dados[ticker] = df
            
        return dados

    def descobrir_clusters(self, dados_dit):
        """
        Usa Machine Learning Não-Supervisionado (DBSCAN) para agrupar 
        moedas que tiveram retornos semelhantes nos últimos 30 dias.
        """
        print("Fase 1: Agrupando moedas via DBSCAN...")
        df_fechamentos = pd.DataFrame({ticker: df['close'] for ticker, df in dados_dit.items()})
        
        # Calcula os retornos percentuais
        retornos = df_fechamentos.pct_change().dropna()
        
        # Transpõe para que os tickers sejam as linhas (amostras) e as datas sejam as features
        X = retornos.T
        X_scaled = StandardScaler().fit_transform(X)
        
        # O eps e min_samples controlam a rigidez do agrupamento
        dbscan = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
        clusters = dbscan.fit_predict(X_scaled)
        
        grupos = {}
        for ticker, cluster_id in zip(retornos.columns, clusters):
            if cluster_id != -1: # -1 significa ruído (moeda sem correlação)
                if cluster_id not in grupos:
                    grupos[cluster_id] = []
                grupos[cluster_id].append(ticker)
                
        return grupos

    def testar_cointegracao_pares(self, grupos, dados_dit):
        """
        Testa todas as combinações dentro de um cluster para confirmar 
        a estacionariedade do Spread (se o elástico volta).
        """
        print("Fase 2: Aplicando Teste de Dickey-Fuller Aumentado (Cointegração)...")
        pares_aprovados = []
        
        for cluster_id, tickers in grupos.items():
            if len(tickers) < 2:
                continue
                
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    ticker_a = tickers[i]
                    ticker_b = tickers[j]
                    
                    serie_a = dados_dit[ticker_a]['close']
                    serie_b = dados_dit[ticker_b]['close']
                    
                    # O teste p-value diz a probabilidade do par não ser cointegrado
                    score, p_value, _ = coint(serie_a, serie_b)
                    
                    if p_value < self.p_value_threshold:
                        pares_aprovados.append((ticker_a, ticker_b))
                        print(f"Par aprovado: {ticker_a} x {ticker_b} (p-value: {p_value:.4f})")
                        
        return pares_aprovados

    def criar_candle_sintetico(self, df_a, df_b):
        """
        Funde as duas moedas para criar o Ativo Sintético (Spread), 
        permitindo o uso dos 34 indicadores no ta_indicators.py.
        """
        # Equações vetoriais para fusão de liquidez e distorção
        df_sintetico = pd.DataFrame(index=df_a.index)
        df_sintetico['open'] = df_a['open'] - df_b['open']
        df_sintetico['high'] = df_a['high'] - df_b['low']
        df_sintetico['low'] = df_a['low'] - df_b['high']
        df_sintetico['close'] = df_a['close'] - df_b['close']
        df_sintetico['volume'] = (df_a['volume'] + df_b['volume']) / 2
        
        return df_sintetico

    def treinar_modelo_par(self, par, df_sintetico):
        """
        Treina o LightGBM usando o Spread sintético enriquecido com os indicadores.
        """
        print(f"Fase 3 & 4: Criando Sintético e Treinando IA para {par[0]} x {par[1]}...")
        
        # Aplica os 34 indicadores da sua arquitetura no Spread
        # df_sintetico = adicionar_todos_indicadores(df_sintetico)
        
        # Simulação da criação de um alvo (Target) para o treino (Reversão à média)
        # 1 = Comprar o Spread, -1 = Vender o Spread, 0 = Ficar de fora
        df_sintetico['target'] = np.where(df_sintetico['close'].shift(-1) > df_sintetico['close'], 1, 0)
        df_sintetico.dropna(inplace=True)
        
        features = ['open', 'high', 'low', 'close', 'volume'] # Na prática, os 34 indicadores
        X = df_sintetico[features]
        y = df_sintetico['target']
        
        # Configuração do LightGBM otimizada para baixo consumo de RAM na VPS
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X, label=y)
        modelo = lgb.train(params, train_data, num_boost_round=100)
        
        # Salva o cérebro da arbitragem em disco para o radar ler
        nome_arquivo = f"{self.modelos_dir}/arb_{par[0]}_{par[1]}.pkl"
        joblib.dump(modelo, nome_arquivo)
        print(f"Modelo salvo: {nome_arquivo}")

    async def executar_pipeline(self):
        """
        Orquestra a esteira sequencial (acionada pelo main.py).
        """
        dados_dit = await self.obter_dados_historicos()
        grupos = self.descobrir_clusters(dados_dit)
        pares_validos = self.testar_cointegracao_pares(grupos, dados_dit)
        
        for par in pares_validos:
            ticker_a, ticker_b = par
            df_sintetico = self.criar_candle_sintetico(dados_dit[ticker_a], dados_dit[ticker_b])
            self.treinar_modelo_par(par, df_sintetico)
            
        print("Treinamento de Arbitragem Estatística finalizado. Liberando RAM.")

# Bloco para teste isolado
if __name__ == "__main__":
    treinador_arb = TreinadorArbitragem(dias_historico=30)
    asyncio.run(treinador_arb.executar_pipeline())
