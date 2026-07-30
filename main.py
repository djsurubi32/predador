import multiprocessing
import asyncio
import logging
import sys
import time
import psutil

# Importação dos microserviços isolados
from treinador import main as treinador_main
from analisador import RadarCore
from gerenciador import GerenciadorRiscoAutonomo
from radar_pares import RadarArbitragem
from treinador_arbitragem import TreinadorArbitragem
from config import Config

# Configuração do Orquestrador
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MAESTRO] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def verificar_uso_ram():
    """Retorna o percentual de uso da memória RAM da VPS."""
    return psutil.virtual_memory().percent

def run_treinadores_sequenciais():
    """Microserviço 1: Esteira de Inteligência Artificial (Heavy CPU - Sequencial)"""
    logging.info("🧠 Iniciando microserviço: TREINADORES (Direcional -> Arbitragem)")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    async def esteira():
        while True:
            logging.info("🔄 [ESTEIRA] Fase 1: Iniciando Treinamento Direcional...")
            try:
                # Importante: O treinador_main() não pode ter um "while True" infinito dentro dele,
                # ele deve executar um ciclo completo de treino e finalizar para liberar a fila.
                await treinador_main()
            except Exception as e:
                logging.error(f"❌ Erro no Treinador Direcional: {e}")
            
            logging.info("🔄 [ESTEIRA] Fase 2: Iniciando Treinamento de Arbitragem...")
            try:
                treinador_arb = TreinadorArbitragem(dias_historico=Config.ARB_DIAS_HISTORICO)
                await treinador_arb.executar_pipeline()
            except Exception as e:
                logging.error(f"❌ Erro no Treinador de Arbitragem: {e}")
            
            horas = getattr(Config, 'HORAS_RETREINO', 24)
            logging.info(f"💤 [ESTEIRA] Ciclo completo. Hibernando por {horas} horas...")
            await asyncio.sleep(horas * 3600)

    try:
        asyncio.run(esteira())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal na Esteira de Treinadores: {e}")

def run_analisador():
    """Microserviço 2: Radar de Mercado Direcional (Heavy I/O)"""
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

def run_radar_arbitragem():
    """Microserviço 3: Radar de Mercado Neutro (Pairs Trading)"""
    logging.info("⚖️ Iniciando microserviço: RADAR ARBITRAGEM")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        radar_arb = RadarArbitragem()
        asyncio.run(radar_arb.iniciar_varredura())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal no Radar de Arbitragem: {e}")

def run_gerenciador():
    """Microserviço 4: Cérebro de Risco e Execução Híbrida (Latência Zero)"""
    logging.info("🛡️ Iniciando microserviço: GERENCIADOR HÍBRIDO")
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"❌ Falha fatal no Gerenciador: {e}")

if __name__ == '__main__':
    logging.info("===============================================================")
    logging.info("🚀 SISTEMA PREDADOR QUANTITATIVO V3.1 (HÍBRIDO & DISTRIBUÍDO)")
    logging.info("===============================================================")

    # Criação dos processos independentes
    p_treinadores = multiprocessing.Process(target=run_treinadores_sequenciais, name="Processo-Treinadores")
    p_analisador = multiprocessing.Process(target=run_analisador, name="Processo-Analisador")
    p_radar_arb = multiprocessing.Process(target=run_radar_arbitragem, name="Processo-Radar-Arbitragem")
    p_gerenciador = multiprocessing.Process(target=run_gerenciador, name="Processo-Gerenciador")

    # Ordem de ignição faseada
    logging.info("⏳ Ligando Analisador Direcional e Radar de Arbitragem...")
    p_analisador.start()
    p_radar_arb.start()
    time.sleep(20) 

    p_gerenciador.start()
    time.sleep(5) 

    uso_ram = verificar_uso_ram()
    if uso_ram < 85.0:
        logging.info(f"📊 RAM estável ({uso_ram}%). Iniciando Esteira de Treinadores...")
        p_treinadores.start()
    else:
        logging.warning(f"⚠️ RAM crítica na VPS ({uso_ram}%). A Esteira aguardará alívio para iniciar.")

    try:
        # Cão de guarda (Watchdog)
        while True:
            time.sleep(30)

            if not p_analisador.is_alive():
                logging.warning("⚠️ O processo ANALISADOR caiu. Reiniciando...")
                p_analisador = multiprocessing.Process(target=run_analisador, name="Processo-Analisador")
                p_analisador.start()
                time.sleep(10)

            if not p_radar_arb.is_alive():
                logging.warning("⚠️ O processo RADAR ARBITRAGEM caiu. Reiniciando...")
                p_radar_arb = multiprocessing.Process(target=run_radar_arbitragem, name="Processo-Radar-Arbitragem")
                p_radar_arb.start()
                time.sleep(10)

            if not p_gerenciador.is_alive():
                logging.warning("⚠️ O processo GERENCIADOR caiu. Reiniciando...")
                p_gerenciador = multiprocessing.Process(target=run_gerenciador, name="Processo-Gerenciador")
                p_gerenciador.start()
                time.sleep(5)

            if not p_treinadores.is_alive():
                uso_ram = verificar_uso_ram()
                if uso_ram < 85.0:
                    logging.warning(f"⚠️ O processo TREINADORES caiu/fila. RAM em {uso_ram}%. (Re)iniciando...")
                    p_treinadores = multiprocessing.Process(target=run_treinadores_sequenciais, name="Processo-Treinadores")
                    p_treinadores.start()
                else:
                    logging.warning(f"⚠️ TREINADORES inativo. RAM crítica ({uso_ram}%).")

    except KeyboardInterrupt:
        # Morte Limpa
        logging.info("🛑 Comando de paragem recebido. A desligar todos os motores de forma segura...")

        if p_treinadores.is_alive(): p_treinadores.terminate()
        if p_analisador.is_alive(): p_analisador.terminate()
        if p_radar_arb.is_alive(): p_radar_arb.terminate()
        if p_gerenciador.is_alive(): p_gerenciador.terminate()

        p_treinadores.join()
        p_analisador.join()
        p_radar_arb.join()
        p_gerenciador.join()

        logging.info("✅ Sistema integralmente encerrado.")
        sys.exit(0)
