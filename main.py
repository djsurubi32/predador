import sys
import os
import time
import logging
import asyncio
import threading
import ccxt
from http.server import BaseHTTPRequestHandler, HTTPServer

from config import Config
from treinador import MotorTreinamento
from analisador import AnalisadorRadar
from executor import EngineExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Escudo Anti-Crash (Keep-Alive)
class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Predador Quantitativo Online")

    def log_message(self, format, *args):
        # Desativa o log de requisições web para não poluir o terminal
        pass

def run_keep_alive():
    try:
        server = HTTPServer(('0.0.0.0', 8080), KeepAliveHandler)
        logging.info("🛡️ Escudo Anti-Crash (Healthcheck) ativado na porta 8080")
        server.serve_forever()
    except Exception as e:
        logging.error(f"Erro no Escudo Anti-Crash: {e}")

def run_treinador_loop():
    treinador = MotorTreinamento()
    while True:
        try:
            treinador.iniciar_ciclo_treinamento()
            time.sleep(Config.HORAS_RETREINO * 3600)
        except Exception as e:
            logging.error(f"Erro Crítico no Loop do Treinador: {e}")
            time.sleep(60)

def run_radar_loop(public_exchange):
    # É fundamental criar um novo event loop para a thread do Radar assíncrono
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    radar = AnalisadorRadar(public_exchange)
    try:
        loop.run_until_complete(radar.loop_radar())
    except Exception as e:
        logging.error(f"Erro Crítico no Loop do Radar: {e}")

def main():
    logging.info("A ativar o Escudo Anti-Crash (Keep-Alive)...")
    threading.Thread(target=run_keep_alive, daemon=True, name="Thread-KeepAlive").start()

    # 1. Inicializa a conexão pública da corretora para o Radar e Executor
    public_exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    # 2. Inicia o Cérebro Institucional (Treinador) em uma thread dedicada e síncrona
    logging.info("A iniciar a thread do Treinador de IA...")
    threading.Thread(target=run_treinador_loop, daemon=True, name="Thread-Treinador").start()

    # 3. Pequena pausa para garantir que os modelos iniciais de IA sejam validados
    time.sleep(5)

    # 4. Inicia o Radar (Analisador) em uma thread assíncrona dedicada
    logging.info("A iniciar a thread do Analisador (Radar)...")
    threading.Thread(target=run_radar_loop, args=(public_exchange,), daemon=True, name="Thread-Radar").start()

    # 5. Pequena pausa para garantir que o Radar limpe o banco e comece a gravar os sinais limpos
    time.sleep(3)

    # 6. Inicia o Executor de Ordens na thread principal (Main Thread)
    logging.info("A iniciar o Executor de Ordens (Main Thread)...")
    executor = EngineExecutor()
    try:
        asyncio.run(executor.start_execution_loop())
    except KeyboardInterrupt:
        logging.info("🛑 Sistema encerrado pelo usuário.")

if __name__ == '__main__':
    # Otimização de plataforma para Windows (se rodar fora da VPS)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
