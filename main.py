import sys
import os
import time
import logging
import asyncio
import threading
import ccxt
from http.server import BaseHTTPRequestHandler, HTTPServer

from config import Config
from analisador import iniciar_motores_ia
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

def main():
    logging.info("A ativar o Escudo Anti-Crash (Keep-Alive)...")
    threading.Thread(target=run_keep_alive, daemon=True, name="Thread-KeepAlive").start()

    # 1. Inicializa a conexão pública da corretora para a IA estudar os gráficos
    public_exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    # 2. Dispara a nova arquitetura do Analisador (Treinador + Radar em paralelo)
    iniciar_motores_ia(public_exchange)

    # 3. Pequena pausa para garantir que os modelos de IA e o banco de dados carreguem
    time.sleep(3)

    # 4. Inicia o Executor de Ordens no loop principal do sistema
    logging.info("A iniciar a thread do Executor de Ordens...")
    executor = EngineExecutor()
    try:
        asyncio.run(executor.start_execution_loop())
    except KeyboardInterrupt:
        logging.info("🛑 Executor encerrado pelo usuário.")

if __name__ == '__main__':
    # Otimização de plataforma para Windows (se rodar fora da VPS)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
