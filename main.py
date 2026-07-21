import sys
import time
import logging
import asyncio
import threading

from config import Config
from treinador import MotorTreinamento
from analisador import RadarCore  # CORREÇÃO: Nome e importação corretos
from executor import EngineExecutor
from keep_alive import keep_alive # CORREÇÃO: Utilizando o seu módulo modularizado

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_treinador_loop():
    """Thread Síncrona: Treina os modelos de IA e hiberna."""
    treinador = MotorTreinamento()
    while True:
        try:
            treinador.iniciar_ciclo_treinamento()
            time.sleep(Config.HORAS_RETREINO * 3600)
        except Exception as e:
            logging.error(f"Erro Crítico no Loop do Treinador: {e}")
            time.sleep(60)

def run_radar_loop():
    """Thread Assíncrona: Varre o mercado 24/7 alimentando o banco de dados (Sinais)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # O RadarCore agora usa o seu próprio pool otimizado de conexões HTTP (Sem injeção externa)
    radar = RadarCore() 
    try:
        # CORREÇÃO: Chamando o método real do analisador
        loop.run_until_complete(radar.scan_market()) 
    except Exception as e:
        logging.error(f"Erro Crítico no Loop do Radar: {e}")

def main():
    logging.info("🛡️ Iniciando ecossistema Predador 2.0...")
    
    # 1. Ativa o Escudo Anti-Crash (Healthcheck da VPS)
    keep_alive()

    # 2. Inicia o Cérebro Institucional (Treinador de IA)
    logging.info("🧠 Ligando Motor de Machine Learning (Treinador)...")
    threading.Thread(target=run_treinador_loop, daemon=True, name="Treinador").start()

    # Breve pausa para garantir que o modelo inicie a alocação de memória com segurança
    time.sleep(5)

    # 3. Inicia o Motor de Varredura (Radar)
    logging.info("📡 Ligando Motor de Varredura Quantitativa (Radar)...")
    threading.Thread(target=run_radar_loop, daemon=True, name="Radar").start()

    time.sleep(3)

    # 4. Inicia o Motor de Execução Institucional (Na Main Thread)
    logging.info("⚡ Ligando Motor de Execução e Catraca Elástica...")
    executor = EngineExecutor()
    try:
        asyncio.run(executor.start_execution_loop())
    except KeyboardInterrupt:
        logging.info("🛑 Sistema Predador encerrado pelo usuário.")

if __name__ == '__main__':
    # Otimização assíncrona mandatória para ambientes Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
