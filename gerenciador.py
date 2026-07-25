import sys
import os
import asyncio
import logging
import time
import numpy as np
from config import Config
from executor import EngineExecutor, TelegramLogger

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GERENCIADOR] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GerenciadorRiscoAutonomo:
    def __init__(self):
        self.executor = EngineExecutor()
        self.db = self.executor.db
        
        # Estado interno por símbolo
        # Guarda: entry_price, original_sl, risk_distance, current_sl, current_tp, side, qty, be_ativado
        self.posicoes_monitoradas = {}
        
        self.last_summary_time = time.time()
        self.pnl_realizado_acumulado = 0.0

    def _calcular_risco_e_alvos(self, entry: float, sl: float, side: str) -> dict:
        """
        Calcula a distância de risco (1R) e define o Take Profit exatamente em 2R.
        Também prepara o preço de Break-Even.
        """
        if side.upper() in ['BUY', 'LONG']:
            risk_distance = abs(entry - sl)
            tp = entry + (risk_distance * 2.0)          # 2R
            be_price = entry + (risk_distance * 0.05)   # BE + 5% do risco (cobre taxas)
        else:
            risk_distance = abs(sl - entry)
            tp = entry - (risk_distance * 2.0)          # 2R
            be_price = entry - (risk_distance * 0.05)   # BE + 5% do risco

        return {
            "risk_distance": risk_distance,
            "tp": tp,
            "be_price": be_price
        }

    async def sincronizar_posicoes_abertas(self):
        """
        Busca todas as posições abertas na corretora e sincroniza o estado interno.
        Só gerencia o que já está aberto. Não abre nada.
        """
        try:
            positions = await self.executor.execution.get_current_positions(self.db)
        except Exception as e:
            logging.error(f"Erro ao buscar posições na corretora: {e}")
            return

        symbols_ativos = set()

        for p in positions:
            symbol = p.get('symbol')
            if not symbol:
                continue

            symbols_ativos.add(symbol)

            side = p.get('side', '').upper()
            entry = float(p.get('entryPrice', p.get('price', 0.0)))
            qty = float(p.get('contracts', p.get('amount', p.get('size', 0.0))))
            current_sl = float(p.get('stopLoss', p.get('sl', 0.0)))
            current_tp = float(p.get('takeProfit', p.get('tp', 0.0)))
            net_pnl = float(p.get('netPnl', 0.0))

            # Se a posição ainda não está sendo monitorada, inicializa o estado 2:1
            if symbol not in self.posicoes_monitoradas:
                # Caso a corretora não tenha SL definido, usamos um fallback conservador
                if current_sl <= 0:
                    # Fallback de 1.5% de risco se não houver SL
                    if side in ['BUY', 'LONG']:
                        current_sl = entry * 0.985
                    else:
                        current_sl = entry * 1.015

                alvos = self._calcular_risco_e_alvos(entry, current_sl, side)

                self.posicoes_monitoradas[symbol] = {
                    "entry": entry,
                    "original_sl": current_sl,
                    "risk_distance": alvos["risk_distance"],
                    "current_sl": current_sl,
                    "current_tp": alvos["tp"],
                    "be_price": alvos["be_price"],
                    "side": side,
                    "qty": qty,
                    "be_ativado": False,
                    "max_pnl": net_pnl,
                    "min_pnl": net_pnl
                }

                # Força o TP para 2R na corretora (se a API permitir)
                try:
                    await self.executor.execution.modify_position_tp_sl(
                        symbol=symbol,
                        take_profit=alvos["tp"],
                        stop_loss=current_sl
                    )
                    logging.info(f"[{symbol}] TP forçado para 2R | TP: {alvos['tp']:.5f} | SL: {current_sl:.5f}")
                except Exception as e:
                    logging.warning(f"[{symbol}] Não foi possível modificar TP/SL na corretora: {e}")

            else:
                # Atualiza extremos de PnL
                mon = self.posicoes_monitoradas[symbol]
                if net_pnl > mon["max_pnl"]:
                    mon["max_pnl"] = net_pnl
                if net_pnl < mon["min_pnl"]:
                    mon["min_pnl"] = net_pnl

        # Remove do monitoramento posições que já não existem mais
        symbols_para_remover = [s for s in self.posicoes_monitoradas if s not in symbols_ativos]
        for s in symbols_para_remover:
            del self.posicoes_monitoradas[s]

    async def gerenciar_posicoes_2r(self):
        """
        Lógica principal de gerenciamento 2:1 por posição individual.
        - Move SL para Break-Even + buffer quando atinge +1R
        - Nunca fecha a cesta inteira
        - Deixa o trade fluir até o 2R ou ser estopado no BE
        """
        if not self.posicoes_monitoradas:
            return

        try:
            positions = await self.executor.execution.get_current_positions(self.db)
        except Exception as e:
            logging.error(f"Erro ao buscar posições para gerenciamento 2R: {e}")
            return

        pos_dict = {p.get('symbol'): p for p in positions if p.get('symbol')}

        for symbol, mon in list(self.posicoes_monitoradas.items()):
            if symbol not in pos_dict:
                continue

            p = pos_dict[symbol]
            net_pnl = float(p.get('netPnl', 0.0))
            entry = mon["entry"]
            risk = mon["risk_distance"]
            side = mon["side"]
            qty = mon["qty"]

            # Valor de 1R em dinheiro (aproximado)
            # Usamos o notional / leverage implícito, mas o mais confiável é o PnL atual vs distância de preço
            current_price = float(p.get('markPrice', p.get('price', entry)))
            
            if side in ['BUY', 'LONG']:
                pnl_em_r = (current_price - entry) / risk if risk > 0 else 0.0
            else:
                pnl_em_r = (entry - current_price) / risk if risk > 0 else 0.0

            # === MOVE PARA BREAK-EVEN quando atinge +1R ===
            if not mon["be_ativado"] and pnl_em_r >= 1.0:
                novo_sl = mon["be_price"]
                
                try:
                    await self.executor.execution.modify_position_tp_sl(
                        symbol=symbol,
                        take_profit=mon["current_tp"],
                        stop_loss=novo_sl
                    )
                    mon["current_sl"] = novo_sl
                    mon["be_ativado"] = True
                    
                    msg = (
                        f"🔒 [{symbol}] BREAK-EVEN ATIVADO (1R atingido)\n"
                        f"Side: {side} | Entry: {entry:.5f}\n"
                        f"Novo SL (BE+): {novo_sl:.5f}\n"
                        f"TP permanece em 2R: {mon['current_tp']:.5f}\n"
                        f"PnL atual: ${net_pnl:+.2f} ({pnl_em_r:.2f}R)"
                    )
                    await TelegramLogger.send(msg)
                    logging.info(f"[{symbol}] SL movido para Break-Even + buffer | Novo SL: {novo_sl:.5f}")
                    
                except Exception as e:
                    logging.error(f"[{symbol}] Falha ao mover SL para BE: {e}")

    async def relatorio_periodico(self):
        """
        Relatório limpo a cada 10 minutos — sem catraca, só status real das posições.
        """
        now = time.time()
        if now - self.last_summary_time < 600:
            return

        self.last_summary_time = now

        if not self.posicoes_monitoradas:
            return

        try:
            positions = await self.executor.execution.get_current_positions(self.db)
        except Exception:
            return

        total_net_pnl = sum(float(p.get('netPnl', 0.0)) for p in positions)
        total_margin_estimada = 0.0

        linhas = []
        for p in positions:
            symbol = p.get('symbol')
            net_pnl = float(p.get('netPnl', 0.0))
            mon = self.posicoes_monitoradas.get(symbol, {})
            
            be_status = "BE ATIVO" if mon.get("be_ativado") else "SL Original"
            linhas.append(
                f"• {symbol} | {mon.get('side', '?')} | PnL: ${net_pnl:+.2f} | {be_status}"
            )

            # Estimativa grosseira de margem (só para relatório)
            qty = float(p.get('contracts', p.get('amount', 0.0)))
            price = float(p.get('entryPrice', 0.0))
            total_margin_estimada += (qty * price) / 20.0   # assume \~20x média

        modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
        banca_inicial = getattr(Config, 'BANCA_INICIAL', 100.0)
        saldo_estimado = banca_inicial + self.pnl_realizado_acumulado + total_net_pnl

        resumo = (
            f"⏱️ <b>RAIO-X DAS POSIÇÕES (10 min)</b> [{modo_texto}]\n\n"
            f"🔹 Posições Ativas: {len(positions)}\n"
            f"🔹 PnL Flutuante Total: ${total_net_pnl:+.2f}\n"
            f"🔹 Saldo Estimado: ${saldo_estimado:.2f}\n\n"
            f"<b>Detalhamento:</b>\n" + "\n".join(linhas)
        )
        await TelegramLogger.send(resumo)
        logging.info("Relatório periódico de posições enviado.")

    async def loop_agente_autonomo(self):
        logging.info("🧠 GERENCIADOR DE RISCO 2:1 ONLINE — Apenas gerenciamento de posições abertas")
        
        while True:
            try:
                await self.sincronizar_posicoes_abertas()
                await self.gerenciar_posicoes_2r()
                await self.relatorio_periodico()
            except Exception as e:
                logging.error(f"Erro crítico no loop do gerenciador: {e}")
            finally:
                await asyncio.sleep(3)   # 3 segundos é suficiente e reduz carga

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador de risco 2:1 desligado pelo operador.")
