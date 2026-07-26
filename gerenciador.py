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

    async def executar_novas_entradas(self):
        """
        [NOVO MÓDULO] Lê a Mesa do Leilão (sinais do Analisador) e executa as
        ordens obedecendo o gerenciamento de risco e tamanho de posição.
        """
        try:
            sinais = await self.db.get_elite_signals()
            if not sinais:
                return

            # 1. Trava de Segurança Global (Max Trades)
            positions = await self.executor.execution.get_current_positions(self.db)
            max_trades = getattr(Config, 'MAX_OPEN_TRADES', 5)
            if len(positions) >= max_trades:
                return

            simbolos_abertos = [p.get('symbol') for p in positions]

            for sinal in sinais:
                symbol = sinal['symbol']
                direction = sinal['direction']
                entry = sinal['price']
                atr = sinal.get('current_atr', entry * 0.01)

                # Impede abrir duas ordens na mesma moeda
                if symbol in simbolos_abertos:
                    continue

                # Verifica Cooldown (impede entrar na mesma moeda após tomar Stop)
                cooldown = await self.db.get_cooldown(symbol)
                if time.time() < cooldown:
                    continue

                # 2. Cálculo do Stop Loss Institucional (Baseado na Volatilidade/ATR)
                if direction in ['BUY', 'LONG']:
                    sl = entry - (atr * 1.5) # Stop fica a 1.5x a volatilidade da moeda
                else:
                    sl = entry + (atr * 1.5)

                alvos = self._calcular_risco_e_alvos(entry, sl, direction)
                tp = alvos['tp']

                # 3. Position Sizing Matemático (Risco em Dólares)
                # Define o risco máximo fixo por trade (Padrão: $5 dólares se não existir no Config)
                risco_usd = getattr(Config, 'RISCO_POR_TRADE_USD', 5.0)

                risk_distance_price = abs(entry - sl)
                if risk_distance_price == 0:
                    continue

                # Fórmula Clássica de Lote: Qty = Risco Financeiro / Distância do Stop
                qty = risco_usd / risk_distance_price

                # 4. Empacota a ordem para o Roteador do Executor
                order_packet = {
                    'symbol': symbol,
                    'direction': direction,
                    'qty': qty,
                    'current_price': entry,
                    'tp': tp,
                    'sl': sl
                }

                logging.info(f"⚡ ORDEM DE DISPARO: {symbol} | {direction} | Qty: {qty:.4f} | Risco: ${risco_usd:.2f}")
                sucesso = await self.executor.order_router_inbound(order_packet)

                if sucesso:
                    # Notifica no Telegram
                    msg = (
                        f"🎯 <b>NOVA POSIÇÃO ABERTA</b>\n"
                        f"Moeda: {symbol}\n"
                        f"Direção: {direction}\n"
                        f"Entrada: {entry:.5f}\n"
                        f"Take Profit (2R): {tp:.5f}\n"
                        f"Stop Loss (ATR): {sl:.5f}\n"
                        f"Score IA: {sinal.get('score', 0):.1f}/10"
                    )
                    await TelegramLogger.send(msg)

                    simbolos_abertos.append(symbol)
                    if len(simbolos_abertos) >= max_trades:
                        break # Encheu o balde de ordens ativas

            # Limpa a mesa de leilão após processar para evitar repetições
            await self.db.clear_elite_signals()

        except Exception as e:
            logging.error(f"Erro no módulo de execução de novas entradas: {e}")

    async def sincronizar_posicoes_abertas(self):
        """
        Busca todas as posições abertas na corretora e sincroniza o estado interno.
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

            if symbol not in self.posicoes_monitoradas:
                if current_sl <= 0:
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
                mon = self.posicoes_monitoradas[symbol]
                if net_pnl > mon["max_pnl"]:
                    mon["max_pnl"] = net_pnl
                if net_pnl < mon["min_pnl"]:
                    mon["min_pnl"] = net_pnl

        symbols_para_remover = [s for s in self.posicoes_monitoradas if s not in symbols_ativos]
        for s in symbols_para_remover:
            del self.posicoes_monitoradas[s]

    async def gerenciar_posicoes_2r(self):
        """
        Lógica principal de gerenciamento 2:1 por posição individual.
        - Move SL para Break-Even + buffer quando atinge +1R
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

            current_price = float(p.get('markPrice', p.get('price', entry)))

            if side in ['BUY', 'LONG']:
                pnl_em_r = (current_price - entry) / risk if risk > 0 else 0.0
            else:
                pnl_em_r = (entry - current_price) / risk if risk > 0 else 0.0

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

        linhas = []
        for p in positions:
            symbol = p.get('symbol')
            net_pnl = float(p.get('netPnl', 0.0))
            mon = self.posicoes_monitoradas.get(symbol, {})

            be_status = "BE ATIVO" if mon.get("be_ativado") else "SL Original"
            linhas.append(
                f"• {symbol} | {mon.get('side', '?')} | PnL: ${net_pnl:+.2f} | {be_status}"
            )

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
        logging.info("🧠 GERENCIADOR DE RISCO 2:1 ONLINE — Módulo de Abertura e Gestão Ativos")

        while True:
            try:
                # 1. Lê a mesa de leilão e compra/vende
                await self.executar_novas_entradas()

                # 2. Atualiza a memória com o que foi executado
                await self.sincronizar_posicoes_abertas()

                # 3. Protege as posições ativas
                await self.gerenciar_posicoes_2r()

                # 4. Envia relatórios ao Telegram
                await self.relatorio_periodico()

            except Exception as e:
                logging.error(f"Erro crítico no loop do gerenciador: {e}")
            finally:
                await asyncio.sleep(3)   # Intervalo suave

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador de risco 2:1 desligado pelo operador.")
