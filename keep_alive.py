from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import os
import logging

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Predador Quantitativo 10/10 Online e Operando.")
        
    def log_message(self, format, *args):
        # Desativa o spam de logs do servidor web
        pass

def run_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), PingHandler)
    logging.info(f"🛡️ Escudo Anti-Crash (Healthcheck) ativado na porta {port}")
    server.serve_forever()

def keep_alive():
    t = Thread(target=run_server)
    t.daemon = True
    t.start()
