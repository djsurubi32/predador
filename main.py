import sys
import asyncio
import logging
import time
from threading import Thread

from config import Config
from treinador import MotorTreinamento
from analisador import RadarCore
from executor import EngineExecutor

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORQUESTRADOR] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class OrquestradorRobo:
    def __init__(self):
        self.treinador = MotorTreinamento()
        self.analisador = RadarCore()
        self.executor = EngineExecutor()

    def _loop_treinamento_background(self):
        """
        Thread dedicada para o processamento matemático pesado da IA.
        Consome as 2 vCPUs de forma isolada sem atrasar o envio de ordens.
        """
        logging.info("🧠 Thread de segundo plano do Treinador de IA inicializada.")
        while True:
            try:
                self.treinador.iniciar_ciclo_treinamento()
                logging.info(f"💤 Treinamento concluído. O cérebro vai hibernar por {Config.HORAS_RETREINO} horas.")
                time.sleep(Config.HORAS_RETREINO * 3600)
            except Exception as e:
                logging.error(f"❌ Erro na Thread do Treinador: {e}")
                time.sleep(60)

    async def iniciar_sistema(self):
        logging.info("🚀 Inicializando ecossistema do Predador Bot na Discloud...")

        # Inicializa a thread de treinamento em background
        thread_treino = Thread(target=self._loop_treinamento_background, daemon=True)
        thread_treino.start()

        await asyncio.sleep(2)

        # Concorrência pura: Analisador e Executor dividem o mesmo espaço de memória RAM
        logging.info("📡 Acoplando motores assíncronos do Analisador e do Executor de Ordens...")
        try:
            await asyncio.gather(
                self.analisador.scan_market(),
                self.executor.start_execution_loop()
            )
        except Exception as e:
            logging.error(f"💥 Falha crítica no núcleo assíncrono do robô: {e}")

if __name__ == "__main__":
    orquestrador = OrquestradorRobo()
    try:
        asyncio.run(orquestrador.iniciar_sistema())
    except KeyboardInterrupt:
        logging.info("🛑 Sistema encerrado pelo usuário.")
