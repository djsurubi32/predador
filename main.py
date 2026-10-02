"""
main.py — Orquestrador do Predador v3.2.

PROCESSOS:
    1. TREINADOR   — reforja modelos a cada HORAS_RETREINO (12h)
    2. ANALISADOR  — emite sinais quando fecha vela de 5m
    3. GERENCIADOR — 1 posicao, risco 1%, time-stop 15 min

IGNICAO FASEADA:
    Analisador + Gerenciador sobem juntos (operam com o que existe).
    Treinador sobe 15s depois (pesado; nao compete na partida).
    Se a RAM estiver > 85%, o treinador fica em espera e tenta a cada ciclo.

WATCHDOG: qualquer processo morto e renascido em ate 30s.

ENCERRAMENTO: Ctrl+C mata a arvore inteira com grace period de 10s.
"""

import sys
import time
import asyncio
import logging
import multiprocessing

import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MAESTRO] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _uso_ram() -> float:
    return psutil.virtual_memory().percent


# =====================================================================
# MICROSERVICO 1 — TREINADOR
# =====================================================================
def run_treinador():
    logging.info("[PROC] Treinador no ar.")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        from treinador import main as treinador_main
        asyncio.run(treinador_main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"[PROC] Treinador morreu: {e}")


# =====================================================================
# MICROSERVICO 2 — ANALISADOR
# =====================================================================
def run_analisador():
    logging.info("[PROC] Analisador no ar.")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        from analisador import main as analisador_main
        asyncio.run(analisador_main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"[PROC] Analisador morreu: {e}")


# =====================================================================
# MICROSERVICO 3 — GERENCIADOR
# =====================================================================
def run_gerenciador():
    logging.info("[PROC] Gerenciador no ar.")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        from gerenciador import GerenciadorRisco
        asyncio.run(GerenciadorRisco().loop())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"[PROC] Gerenciador morreu: {e}")


# =====================================================================
# WATCHDOG
# =====================================================================
class Vigia:
    """Reinicia qualquer processo filho que morrer."""

    ALVOS = {
        "analisador": run_analisador,
        "gerenciador": run_gerenciador,
    }

    def __init__(self):
        self.procs: dict = {}

    def start(self, nome: str, alvo):
        p = multiprocessing.Process(target=alvo, name=f"proc-{nome}", daemon=True)
        p.start()
        self.procs[nome] = p
        logging.info(f"[VIGIA] {nome} iniciado (pid={p.pid}).")

    def checar(self):
        for nome, p in list(self.procs.items()):
            if not p.is_alive():
                logging.warning(f"[VIGIA] {nome} caiu. Renascendo...")
                time.sleep(5)
                self.start(nome, self.ALVOS[nome])
                time.sleep(5)

    def encerrar_todos(self):
        logging.info("[VIGIA] Derrubando processos...")
        for nome, p in self.procs.items():
            if p.is_alive():
                p.terminate()
        for nome, p in self.procs.items():
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        logging.info("[VIGIA] Todos encerrados.")


# =====================================================================
# ORQUESTRADOR
# =====================================================================
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("PREDADOR v3.2 — SCALP MEDIO 5m | 1 POSICAO | GATE 65%")
    logging.info("=" * 60)

    multiprocessing.set_start_method("spawn", force=True)
    vigia = Vigia()

    try:
        # Fase 1: operacao (funciona com modelos ja existentes)
        vigia.start("analisador", run_analisador)
        vigia.start("gerenciador", run_gerenciador)

        # Fase 2: treinador (pesado — sobe com atraso e sob controle de RAM)
        time.sleep(15)
        if _uso_ram() < 85.0:
            vigia.start("treinador", run_treinador)
            vigia.ALVOS["treinador"] = run_treinador
        else:
            logging.warning(
                f"[MAESTRO] RAM {_uso_ram():.0f}% — treinador adiado. "
                "O vigia tentara subi-lo se houver espaco."
            )

        # Loop do vigia
        tentativas_treinador = 0
        while True:
            time.sleep(30)
            vigia.checar()

            # Treinador ainda nao subiu? Tenta quando a RAM permitir
            if "treinador" not in vigia.procs:
                tentativas_treinador += 1
                if tentativas_treinador >= 4 and _uso_ram() < 85.0:
                    vigia.start("treinador", run_treinador)
                    vigia.ALVOS["treinador"] = run_treinador

    except KeyboardInterrupt:
        logging.info("[MAESTRO] Parada solicitada pelo operador.")
        vigia.encerrar_todos()
        logging.info("[MAESTRO] Sistema encerrado integralmente.")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"[MAESTRO] Falha fatal no orquestrador: {e}")
        vigia.encerrar_todos()
        sys.exit(1)
