import time
import logging
from threading import Thread
import asyncio

# Importa o escudo anti-crash (Healthcheck do Railway)
from keep_alive import keep_alive 

# Importa os motores do ecossistema quantitativo
from treinador import main as iniciar_treinador
from analisador import RadarCore
from executor import EngineExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MAIN] - %(message)s')

def rodar_treinador():
    logging.info("A iniciar a thread do Treinador de IA...")
    iniciar_treinador()

def rodar_analisador():
    logging.info("A iniciar a thread do Analisador (Radar)...")
    asyncio.run(RadarCore().scan_market())

def rodar_executor():
    logging.info("A iniciar a thread do Executor de Ordens...")
    asyncio.run(EngineExecutor().start_execution_loop())

if __name__ == "__main__":
    # 1. ATIVA A FANTASIA DE SITE PARA ENGANAR O RAILWAY
    logging.info("A ativar o Escudo Anti-Crash (Keep-Alive)...")
    keep_alive()
    
    # 2. Inicia os motores do robô em paralelo
    Thread(target=rodar_treinador, daemon=True).start()
    
    # Pausa de 5 segundos para garantir que o Treinador crie as pastas e carregue a memória
    time.sleep(5)
    
    Thread(target=rodar_analisador, daemon=True).start()
    
    # Pausa de 2 segundos para evitar sobrecarga de arranque
    time.sleep(2)
    
    # O executor mantém o loop principal vivo na thread primária
    rodar_executor()
