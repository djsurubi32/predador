import multiprocessing
import asyncio
import logging
import sys
import time
import psutil

# Importação dos nossos microserviços isolados
from treinador import main as treinador_main
from analisador import RadarCore
from gerenciador import GerenciadorRiscoAutonomo

# Configuração do Orquestrador
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MAESTRO] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def verificar_uso_ram():
    """Retorna o percentual de uso da memória RAM da VPS."""
    return psutil.virtual_memory().percent

def run_treinador():
    """Microserviço 1: Forja de Inteligência Artificial (Heavy CPU)"""
    logging.info("🧠 Iniciando microserviço: TREINADOR (Machine Learning)")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        # CORREÇÃO: Agora o maestro sabe que o Treinador é um motor assíncrono
        asyncio.run(treinador_main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal no Treinador: {e}")

def run_analisador():
    """Microserviço 2: Radar de Mercado (Heavy I/O)"""
    logging.info("📡 Iniciando microserviço: ANALISADOR (Radar Quantitativo)")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        radar = RadarCore()
        asyncio.run(radar.scan_market())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal no Analisador: {e}")

def run_gerenciador():
    """Microserviço 3: Cérebro de Risco e Execução (Latência Zero)"""
    logging.info("🛡️ Iniciando microserviço: GERENCIADOR (Agente Autônomo e Catraca)")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        # O Gerenciador já importa e instancia o executor.py internamente
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal no Gerenciador: {e}")

if __name__ == '__main__':
    logging.info("===============================================================")
    logging.info("🚀 SISTEMA PREDADOR QUANTITATIVO V2.0 (ARQUITETURA DISTRIBUÍDA)")
    logging.info("===============================================================")

    # Criação dos processos independentes (Isolamento de Memória e CPU)
    p_treinador = multiprocessing.Process(target=run_treinador, name="Processo-Treinador")
    p_analisador = multiprocessing.Process(target=run_analisador, name="Processo-Analisador")
    p_gerenciador = multiprocessing.Process(target=run_gerenciador, name="Processo-Gerenciador")

    # Ordem de ignição faseada para suportar picos de memória da VPS
    logging.info("⏳ Ligando Analisador primeiro para absorver o pico de RAM da IA de NLP...")
    p_analisador.start()
    time.sleep(20)  # Dá 20 segundos para a IA estabilizar na memória

    p_gerenciador.start()
    time.sleep(5)  # Estabilização rápida

    uso_ram = verificar_uso_ram()
    if uso_ram < 85.0:
        logging.info(f"📊 RAM estável ({uso_ram}%). Iniciando Treinador...")
        p_treinador.start()
    else:
        logging.warning(f"⚠️ RAM crítica na VPS ({uso_ram}%). O Treinador aguardará alívio para iniciar.")

    try:
        # O Maestro atua como cão de guarda, reiniciando automaticamente processos mortos
        while True:
            time.sleep(30)

            if not p_analisador.is_alive():
                logging.warning("⚠️ Alerta: O processo ANALISADOR caiu. Reiniciando de forma autônoma...")
                p_analisador = multiprocessing.Process(target=run_analisador, name="Processo-Analisador")
                p_analisador.start()
                time.sleep(20)

            if not p_gerenciador.is_alive():
                logging.warning("⚠️ Alerta: O processo GERENCIADOR caiu. Reiniciando de forma autônoma...")
                p_gerenciador = multiprocessing.Process(target=run_gerenciador, name="Processo-Gerenciador")
                p_gerenciador.start()
                time.sleep(5)

            if not p_treinador.is_alive():
                uso_ram = verificar_uso_ram()
                if uso_ram < 85.0:
                    logging.warning(f"⚠️ O processo TREINADOR caiu ou estava na fila. RAM em {uso_ram}%. (Re)iniciando...")
                    p_treinador = multiprocessing.Process(target=run_treinador, name="Processo-Treinador")
                    p_treinador.start()
                else:
                    logging.warning(f"⚠️ TREINADOR inativo. Aguardando alívio de RAM (Atual: {uso_ram}%) para não travar a VPS.")

    except KeyboardInterrupt:
        # Morte Limpa (Graceful Shutdown) caso prima Ctrl+C no terminal
        logging.info("🛑 Comando de paragem recebido. A desligar todos os motores de forma segura...")

        if p_treinador.is_alive(): p_treinador.terminate()
        if p_analisador.is_alive(): p_analisador.terminate()
        if p_gerenciador.is_alive(): p_gerenciador.terminate()

        p_treinador.join()
        p_analisador.join()
        p_gerenciador.join()

        logging.info("✅ Sistema integralmente encerrado.")
        sys.exit(0)
