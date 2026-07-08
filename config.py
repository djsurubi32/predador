importar os
importar registro
importar ccxt
de digitando importar Lista,Tupla
de dotenv importar carregar_dotenv
registro.configuração básica(
    nível=registro.INFORMAÇÕES,
    formatar='%(asctime)s - %(levelname)s - %(message)s',
    formato de data='%Y-%m-%d %H:%M:%S'
)
lenhador = registro.obterLogger(__nome__)
carregar_dotenv()
definição obter_melhores_moedas(limite:inteiro = 100)-> Lista[str]:
    lenhador.informações(f"🔄 Conectando ao Bybit para buscar o Top{limite}de criptomoedas mais negociadas hoje...")
    tentar:
        intercâmbio = ccxt.bybit({'ativar limite de taxa':Verdadeiro,'opções': {'tipo padrão':'trocar'}})
        intercâmbio.carregar_mercados()
        tickers = intercâmbio.buscar_tickers()
        _validas:Lista[Tupla[str,flutuador]]=[]
        para símbolo,ticker em tickers.Unid():
            mercado = intercâmbio.mercados.pegar(símbolo)
            se mercado e mercado.pegar('linear')e mercado.pegar('citar')== 'USDT' e mercado.pegar('ativo'):
                volume_24h = flutuador(ticker.pegar('quoteVolume',0)ou 0)
                _validas.acrescentar((símbolo,volume_24h))
        _validas.organizar(chave=lambda x:x[1],reverter=Verdadeiro)
        top_ativos =[x[0]para x em _validas[:limite]]
        lenhador.informações(f"✅ Universo atualizado! O robô vai caçar nas{len(top_ativos)}"Cabine mais quentes do dia.")
        retornar top_ativos
    exceto Exceção como e:
        lenhador.erro(f"❌ Erro ao buscar moedas na Bybit:{e}. Acionando lista de segurança.")
        retornar['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','BNB/USDT:USDT']
aula Configuração:
    BYBIT_API_KEY:str = os.getenv("BYBIT_API_KEY","")
    BYBIT_SECRET:str = os.getenv("BYBIT_SECRET","")
    TELEGRAM_TOKEN:str = os.getenv("TELEGRAM_TOKEN","")
    ID_DO_CHAT_DO_TELEGRAM:str = os.getenv("ID_DO_CHAT_DO_TELEGRAM","")
    MODELOS_DIR:str = "modelos_ia"
    #GESTÃO DE RISCO DO BALDE GLOBAL
    LUCRO DA CESTA BASE_ALVO:flutuador = 5,00
    BASE_BASKET_TRAILING_PULLBACK:flutuador = 1,50
    BASE_STOP_BASKET_LOSS:flutuador = -10,00
    MARGEM_DA_CESTA_BASE:flutuador = -15.0
    # PONTO DE EQUILÍBRIO DINÂMICO
    GATILHO_DE_BREAKVEN_BASE:flutuador = 2,50
    LUCRO_DE_EQUILÍBRIO_BASE:flutuador = 1,00
    ENTRADA_DE_PONTUAÇÃO_MIN:flutuador = 3.0
    MAX_PENDING_ORDER_MINUTES:inteiro = 5
    OPERA_CONTA_REAL:booleano = Falso
    BANCA_DEMO_INICIAL:flutuador = 100,0
    KELLY_FRAÇÃO:flutuador = 0,045
    RELAÇÃO RISCO-RECOMPENSA:flutuador = 2.0
    RISCO_MÁXIMO_DE_POSIÇÃO:flutuador = 0,045
    MÁXIMO DE NEGOCIAÇÕES ABERTAS:inteiro = 10
    ALAVANCAGEM:inteiro = 100
    HORIZONTE DE BARREIRA:inteiro = 20
    BARRIER_TP_PCT:flutuador = 1.012
    BARRIER_SL_PCT:flutuador = 0,988
    LIMIAR DE CORRELAÇÃO:flutuador = 0,70
    DURAÇÃO_MÁXIMA_DA_NEGOCIAÇÃO_EM_MINUTOS:inteiro = 240
    # Calibrado para o limite estrito de 2GB de RAM do servidor
    NUM_MOEDAS_OPERACIONAIS:inteiro = 30
    CICLO_SEGUNDOS:inteiro = 1
    TEMPO_ESPERA_HOLD_MINUTOS:inteiro = 5
    PERÍODO DE TEMPO:str = '15m'
    VELAS_TREINAMENTO_ML:inteiro = 1500
    HORAS_RETREINO:inteiro = 4
    NOME_DO_MODELO_NLP:str = 'todos-MiniLM-L6-v2'
    _ATIVOS:Lista[str]=[]
    @método de classe
    definição obter_ativos(cls)-> Lista[str]:
        se não cls._ATIVOS:
            cls._ATIVOS = obter_melhores_moedas(cls.NUM_MOEDAS_OPERACIONAIS)
        retornar cls._ATIVOS
    @método de classe
    definição inicializar_(cls)-> Nenhum:
        se não os.caminho.existe(cls.MODELOS_DIR):
            os.makedirs(cls.MODELOS_DIR)
            lenhador.informações(f"📁 Diretório '{cls.MODELOS_DIR}' preparado com sucesso.")
Configuração.inicializar_()