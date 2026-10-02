"""
gerenciador.py — Gestao de risco e posicoes do Predador v3.2.

PAPEL:
    1. Consome elite_signals (analisador) — so entradas com 1 posicao livre
    2. Sizing: risco = saldo x MAX_POSITION_RISK (1% = $1) — nocional via barreira
       qty = risco_usd / (preco x BARRIER_PCT%)  -> ex.: $1 / 0,20% = $500
    3. SL/TP exatos nas barreiras (+/-BARRIER_PCT%) — payoff 1:1 embutido no rotulo
    4. Time-stop de TIME_STOP_MINUTES (15 min = 3 velas de 5m)
    5. Oraculo (L2+CVD Binance) como filtro final antes de cada entrada
    6. Kill-switch: drawdown diario e sequencia de perdas
    7. Cooldown por simbolo apos stop

CONTRATO COM O EXECUTOR (Passo 7):
    EngineExecutor.order_router_inbound(packet) -> bool
    EngineExecutor.execution.get_current_positions() -> list[dict]
    EngineExecutor.execution.get_equity() -> float
    EngineExecutor.execution.close_position_market(symbol, side, qty) -> bool
    EngineExecutor.modify_position_tp_sl(symbol, take_profit, stop_loss) -> bool
    packet = {symbol, direction, qty, entry_price, tp, sl, barrier_pct, module}
"""

import sys
import time
import asyncio
import logging
import aiosqlite

import numpy as np

from config import Config, BASE_DIR
from executor import EngineExecutor, TelegramLogger
from oraculo import OraculoBinance

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [GERENCIADOR] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SINAL_MAX_IDADE_SEC = 90.0   # ignora sinal mais velho que 1:30
USA_BREAK_EVEN = False       # fase 1: desligado (testar no backtest depois)


class GerenciadorRisco:
    def __init__(self):
        self.executor = EngineExecutor()
        self.oraculo = OraculoBinance()
        self.db_name = str(BASE_DIR / Config.DB_NAME)

        self.posicoes: dict = {}          # symbol -> estado local da posicao
        self.fechando: dict = {}          # symbol -> ts do inicio do fechamento
        self.last_summary = 0.0
        self.pnl_acumulado = 0.0
        self.pnl_dia = 0.0
        self.dia_atual = time.strftime("%Y-%m-%d")
        self.perdas_seg = 0
        self.kill_switch = False

    # ------------------------------------------------------------------
    # ESTADO
    # ------------------------------------------------------------------
    @property
    def saldo(self) -> float:
        """Banca real (equity da conta) + fallback para a configurada."""
        try:
            eq = self.executor.execution.get_equity_sync_if_available()
            if eq and eq > 0:
                return float(eq)
        except Exception:
            pass
        return float(Config.BANCA_INICIAL_USD) + self.pnl_acumulado

    def _reset_dia(self):
        hoje = time.strftime("%Y-%m-%d")
        if hoje != self.dia_atual:
            self.dia_atual = hoje
            self.pnl_dia = 0.0
            self.perdas_seg = 0
            self.kill_switch = False

    def _registrar_resultado(self, pnl: float, symbol: str, motivo: str):
        self.pnl_acumulado += pnl
        self.pnl_dia += pnl
        if pnl < 0:
            self.perdas_seg += 1
        else:
            self.perdas_seg = 0

        asyncio.ensure_future(self._gravar_historico(symbol, pnl, motivo))

        if self.pnl_dia <= -(max(self.saldo, 1.0) * Config.MAX_DAILY_DRAWDOWN):
            self.kill_switch = True
            logging.warning(f"KILL-SWITCH: drawdown diario ${self.pnl_dia:.2f}")
        if self.perdas_seg >= Config.MAX_CONSECUTIVE_LOSSES:
            self.kill_switch = True
            logging.warning(f"KILL-SWITCH: {self.perdas_seg} perdas consecutivas")

    async def _gravar_historico(self, symbol: str, pnl: float, motivo: str):
        try:
            async with aiosqlite.connect(self.db_name, timeout=30) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT, side TEXT, pnl REAL,
                        motivo TEXT, timestamp REAL
                    )
                    """
                )
                await conn.execute(
                    "INSERT INTO history (symbol, side, pnl, motivo, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (symbol, self.posicoes.get(symbol, {}).get("direction", "?"),
                     pnl, motivo, time.time()),
                )
                await conn.commit()
        except Exception as e:
            logging.error(f"Falha ao gravar historico: {e}")

    # ------------------------------------------------------------------
    # LEITURA DE SINAIS
    # ------------------------------------------------------------------
    async def _ler_sinais(self) -> list:
        agora = time.time()
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute(
                """
                SELECT symbol, direction, price, prob, score, barrier_pct, timestamp
                FROM elite_signals
                WHERE timestamp > ?
                ORDER BY prob DESC
                """,
                (agora - SINAL_MAX_IDADE_SEC,),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # NOVAS ENTRADAS
    # ------------------------------------------------------------------
    async def executar_novas_entradas(self):
        if self.kill_switch:
            return
        if len(self.posicoes) >= Config.MAX_OPEN_TRADES:
            return

        sinais = await self._ler_sinais()
        if not sinais:
            return

        positions = await self.executor.execution.get_current_positions()
        abertos_exchange = {p.get("symbol") for p in positions if p.get("symbol")}
        if len(abertos_exchange) >= Config.MAX_OPEN_TRADES:
            return

        for sinal in sinais:
            if len(self.posicoes) >= Config.MAX_OPEN_TRADES:
                break

            symbol = sinal["symbol"]
            direction = sinal["direction"]
            price = float(sinal["price"])
            barrier_pct = float(sinal.get("barrier_pct") or Config.BARRIER_PCT)

            if symbol in self.posicoes or symbol in abertos_exchange:
                continue

            if await self._em_cooldown(symbol):
                continue

            # ---- Oraculo: filtro final de fluxo ----
            aprovado, motivo = await self.oraculo.validar_sinal_institucional(
                symbol, direction
            )
            if not aprovado:
                logging.info(f"Oraculo bloqueou {symbol} {direction}: {motivo}")
                continue

            # ---- Sizing: risco fixo em $, distancia = barreira em % ----
            risco_usd = self.saldo * Config.MAX_POSITION_RISK
            distancia = price * (barrier_pct / 100.0)
            if distancia <= 0:
                continue
            qty = risco_usd / distancia
            if qty <= 0:
                continue

            if direction == "BUY":
                sl = price - distancia
                tp = price + distancia
            else:
                sl = price + distancia
                tp = price - distancia

            packet = {
                "symbol": symbol,
                "direction": direction,
                "qty": qty,
                "entry_price": price,
                "tp": tp,
                "sl": sl,
                "barrier_pct": barrier_pct,
                "module": "directional",
            }

            logging.info(
                f"ENTRY {direction} {symbol} | qty={qty:.6f} "
                f"risco=${risco_usd:.2f} | SL={sl:.6f} TP={tp:.6f} "
                f"(+/-{barrier_pct}%) | Oraculo: {motivo}"
            )

            ok = await self.executor.order_router_inbound(packet)
            if not ok:
                continue

            self.posicoes[symbol] = {
                "direction": direction,
                "entry": price,
                "qty": qty,
                "sl": sl,
                "tp": tp,
                "barrier_dist": distancia,
                "be_price": price,           # BE desativado: preco de entrada
                "be_ativado": True,          # ja "ativado" = nunca move
                "open_time": time.time(),
                "max_pnl_r": 0.0,
            }

            await TelegramLogger.send(
                f"📈 ENTRY {direction} {symbol}\n"
                f"Preço: {price:.6f}\n"
                f"SL: {sl:.6f} | TP: {tp:.6f} (+/-{barrier_pct}%)\n"
                f"Risco: ${risco_usd:.2f} | P(sinal): {sinal['prob']:.1f}%\n"
                f"Time-stop: {Config.TIME_STOP_MINUTES} min\n"
                f"Saldo: ${self.saldo:.2f}\n"
                f"Oráculo: {motivo}"
            )

        await self._limpar_sinais()

    async def _limpar_sinais(self):
        try:
            async with aiosqlite.connect(self.db_name, timeout=30) as conn:
                await conn.execute("DELETE FROM elite_signals")
                await conn.commit()
        except Exception:
            pass

    async def _em_cooldown(self, symbol: str) -> bool:
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            cur = await conn.execute(
                "SELECT release_time FROM cooldowns WHERE symbol = ?", (symbol,)
            )
            row = await cur.fetchone()
        return bool(row and time.time() < row[0])

    async def _set_cooldown(self, symbol: str, minutos: int):
        async with aiosqlite.connect(self.db_name, timeout=30) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cooldowns
                (symbol TEXT PRIMARY KEY, release_time REAL)
                """
            )
            await conn.execute(
                "INSERT OR REPLACE INTO cooldowns VALUES (?, ?)",
                (symbol, time.time() + minutos * 60),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # GESTAO DAS POSICOES ABERTAS
    # ------------------------------------------------------------------
    async def sincronizar_posicoes(self):
        """Alinha estado local com a exchange; detecta fechamentos naturais."""
        try:
            positions = await self.executor.execution.get_current_positions()
        except Exception as e:
            logging.error(f"Falha ao buscar posicoes: {e}")
            return

        ativos_agora = set()
        pos_dict = {p.get("symbol"): p for p in positions if p.get("symbol")}

        for symbol, p in pos_dict.items():
            ativos_agora.add(symbol)
            net_pnl = float(p.get("netPnl", 0.0))

            if symbol not in self.posicoes:
                # Posicao aberta externamente (restart do bot): adota
                entry = float(p.get("entryPrice", 0.0))
                side = str(p.get("side", "")).upper()
                self.posicoes[symbol] = {
                    "direction": "BUY" if side in ("LONG", "BUY") else "SELL",
                    "entry": entry,
                    "qty": float(p.get("contracts", 0.0)),
                    "sl": float(p.get("stopLoss", 0.0) or 0.0),
                    "tp": float(p.get("takeProfit", 0.0) or 0.0),
                    "barrier_dist": 0.0,
                    "be_price": entry,
                    "be_ativado": True,
                    "open_time": time.time(),
                    "max_pnl_r": 0.0,
                }
                logging.warning(f"Posicao {symbol} adotada da exchange (restart).")

            self.posicoes[symbol]["_net_pnl"] = net_pnl

        # Fechadas na exchange (SL/TP tocado ou manual)
        for symbol in list(self.posicoes):
            if symbol not in ativos_agora and symbol not in self.fechando:
                mon = self.posicoes.pop(symbol)
                pnl = mon.get("_net_pnl", 0.0)
                self._registrar_resultado(pnl, symbol, "fechada_na_exchange")
                await self._set_cooldown(symbol, Config.COOLDOWN_POS_LOSS_MIN)
                await TelegramLogger.send(
                    f"🏁 POSICAO ENCERRADA {symbol}\n"
                    f"PnL: ${pnl:+.2f}\nSaldo: ${self.saldo:.2f}"
                )

    async def gerenciar_posicoes(self):
        if not self.posicoes:
            return

        try:
            positions = await self.executor.execution.get_current_positions()
        except Exception:
            return

        pos_dict = {p.get("symbol"): p for p in positions if p.get("symbol")}
        agora = time.time()

        for symbol, mon in list(self.posicoes.items()):
            if symbol in self.fechando:
                if agora - self.fechando[symbol] < 20:
                    continue
                del self.fechando[symbol]

            p = pos_dict.get(symbol)
            if p is None:
                continue  # sincronizar_posicoes ja tratou

            entry = mon["entry"]
            dist = mon["barrier_dist"]
            side = mon["direction"]
            current = float(p.get("markPrice", p.get("price", entry)))
            net_pnl = float(p.get("netPnl", 0.0))
            mon["_net_pnl"] = net_pnl

            # ---- TIME-STOP (15 min) ----
            idade_min = (agora - mon["open_time"]) / 60.0
            if idade_min >= Config.TIME_STOP_MINUTES:
                motivo = f"TIME-STOP ({idade_min:.0f} min)"
                if dist > 0:
                    pnl_r = ((current - entry) / dist) if side == "BUY" \
                        else ((entry - current) / dist)
                    motivo += f" | {pnl_r:+.2f}R"
                await self._fechar(symbol, mon, net_pnl, motivo)
                continue

            # ---- HARD STOP local (rede/WS falhou na exchange) ----
            if dist > 0:
                pnl_r = ((current - entry) / dist) if side == "BUY" \
                    else ((entry - current) / dist)
                mon["max_pnl_r"] = max(mon["max_pnl_r"], pnl_r)

                if pnl_r <= -1.05:  # 5% de folga sobre a barreira negativa
                    await self._fechar(symbol, mon, net_pnl,
                                       f"HARD STOP local ({pnl_r:.2f}R)")
                    continue

                # ---- BREAK-EVEN (fase 2 — desligado) ----
                if USA_BREAK_EVEN and not mon["be_ativado"] and pnl_r >= 1.0:
                    try:
                        await self.executor.modify_position_tp_sl(
                            symbol, take_profit=mon["tp"], stop_loss=mon["be_price"]
                        )
                        mon["be_ativado"] = True
                        mon["sl"] = mon["be_price"]
                        logging.info(f"[{symbol}] BE ativado")
                    except Exception:
                        pass

    async def _fechar(self, symbol: str, mon: dict, net_pnl: float, motivo: str):
        self.fechando[symbol] = time.time()
        self._registrar_resultado(net_pnl, symbol, motivo)

        await TelegramLogger.send(
            f"{'🛑' if net_pnl < 0 else '✅'} {motivo}\n"
            f"{symbol}\nPnL: ${net_pnl:+.2f}\nSaldo: ${self.saldo:.2f}"
        )
        try:
            close_side = "SELL" if mon["direction"] == "BUY" else "BUY"
            await self.executor.execution.close_position_market(
                symbol, close_side, mon["qty"]
            )
        except Exception as e:
            logging.error(f"[{symbol}] Falha ao fechar: {e}")
        finally:
            self.posicoes.pop(symbol, None)
            await self._set_cooldown(symbol, Config.COOLDOWN_POS_LOSS_MIN)

    # ------------------------------------------------------------------
    # RELATORIO
    # ------------------------------------------------------------------
    async def relatorio(self):
        if time.time() - self.last_summary < 300:
            return
        self.last_summary = time.time()
        modo = "TESTNET" if not Config.OPERA_CONTA_REAL else "REAL"
        await TelegramLogger.send(
            f"📊 RAIO-X [{modo}]\n"
            f"Saldo: ${self.saldo:.2f} | PnL dia: ${self.pnl_dia:+.2f}\n"
            f"Posicoes: {len(self.posicoes)}/{Config.MAX_OPEN_TRADES} | "
            f"Perdas seg: {self.perdas_seg}\n"
            f"Kill-switch: {'🚨 ON' if self.kill_switch else 'ok'}"
        )

    # ------------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------------
    async def loop(self):
        logging.info(
            f"GERENCIADOR ONLINE | 1 posicao | risco {Config.MAX_POSITION_RISK:.0%} "
            f"(${(Config.BANCA_INICIAL_USD * Config.MAX_POSITION_RISK):.2f})/trade | "
            f"time-stop {Config.TIME_STOP_MINUTES}min"
        )
        try:
            while True:
                try:
                    self._reset_dia()
                    await self.executar_novas_entradas()
                    await self.sincronizar_posicoes()
                    await self.gerenciar_posicoes()
                    await self.relatorio()
                except Exception as e:
                    logging.error(f"Erro no loop: {e}")
                await asyncio.sleep(2.0)
        finally:
            await self.oraculo.fechar_conexoes()


if __name__ == "__main__":
    try:
        asyncio.run(GerenciadorRisco().loop())
    except KeyboardInterrupt:
        logging.info("Gerenciador desligado.")
