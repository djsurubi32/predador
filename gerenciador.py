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
        try:
            sinais = await self.db.get_elite_signals()
            if not sinais:
                return

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

                if symbol in simbolos_abertos:
                    continue

                cooldown = await self.db.get_cooldown(symbol)
                if time.time() < cooldown:
                    continue

                if direction in ['BUY', 'LONG']:
                    sl = entry - (atr * 1.5)
                else:
                    sl = entry + (atr * 1.5)

                alvos = self._calcular_risco_e_alvos(entry, sl, direction)
                tp = alvos['tp']

                risco_usd = getattr(Config, 'RISCO_POR_TRADE_USD', 5.0)

                risk_distance_price = abs(entry - sl)
                if risk_distance_price == 0:
                    continue

                qty = risco_usd / risk_distance_price

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
                        break 

            await self.db.clear_elite_signals()

        except Exception as e:
            logging.error(f"Erro no módulo de execução de novas entradas: {e}")

    async def sincronizar_posicoes_abertas(self):
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

                # Verifica se a corretora já está com o SL no Break-Even fisicamente
                be_status_inicial = False
                if current_sl > 0:
                    if side in ['BUY', 'LONG'] and current_sl >= alvos["be_price"] * 0.999:
                        be_status_inicial = True
                    elif side in ['SELL', 'SHORT'] and current_sl <= alvos["be_price"] * 1.001:
                        be_status_inicial = True

                self.posicoes_monitoradas[symbol] = {
                    "entry": entry,
                    "original_sl": current_sl,
                    "risk_distance": alvos["risk_distance"],
                    "current_sl": current_sl,
                    "current_tp": alvos["tp"],
                    "be_price": alvos["be_price"],
                    "side": side,
                    "qty": qty,
                    "be_ativado": be_status_inicial,
                    "max_pnl": net_pnl,
                    "min_pnl": net_pnl,
                    "last_net_pnl": net_pnl
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
                mon["last_net_pnl"] = net_pnl
                if net_pnl > mon["max_pnl"]:
                    mon["max_pnl"] = net_pnl
                if net_pnl < mon["min_pnl"]:
                    mon["min_pnl"] = net_pnl

        # Acumula o PnL final no saldo da banca quando a posição é fechada
        symbols_para_remover = [s for s in self.posicoes_monitoradas if s not in symbols_ativos]
        for s in symbols_para_remover:
            pnl_fechamento = self.posicoes_monitoradas[s].get("last_net_pnl", 0.0)
            self.pnl_realizado_acumulado += pnl_fechamento
            logging.info(f"[{s}] Posição encerrada. PnL de ${pnl_fechamento:+.2f} transferido para o saldo da banca.")
            del self.posicoes_monitoradas[s]

    async def gerenciar_posicoes_2r(self):
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

            current_price = float(p.get('markPrice', p.get('price', entry)))
            
            if side in ['BUY', 'LONG']:
                pnl_em_r = (current_price - entry) / risk if risk > 0 else 0.0
            else:
                pnl_em_r = (entry - current_price) / risk if risk > 0 else 0.0

            # --- CATRACA DE SOFTWARE (Hard Take Profit) ---
            if pnl_em_r >= 1.98:
                msg = (
                    f"🏆 <b>ALVO 2R ATINGIDO (VIA SOFTWARE)</b>\n"
                    f"Moeda: {symbol}\n"
                    f"Lucro Final: ${net_pnl:+.2f} ({pnl_em_r:.2f}R)\n"
                    f"Forçando fechamento a mercado para garantir o lucro."
                )
                await TelegramLogger.send(msg)
                logging.info(f"[{symbol}] PnL atingiu {pnl_em_r:.2f}R. Forçando fechamento a mercado.")
                
                try:
                    close_side = "SELL" if side in ["BUY", "LONG"] else "BUY"
                    await self.executor.execution.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=close_side,
                        amount=qty,
                        params={'reduceOnly': True}
                    )
                except Exception as e:
                    logging.error(f"[{symbol}] Falha ao fechar posição a mercado via software: {e}")
                    
                continue 

            # --- GESTÃO DE BREAK-EVEN ---
            current_sl_corretora = float(p.get('stopLoss', p.get('sl', 0.0)))
            ja_no_be = False
            
            if current_sl_corretora > 0:
                if side in ['BUY', 'LONG'] and current_sl_corretora >= mon["be_price"] * 0.999:
                    ja_no_be = True
                elif side in ['SELL', 'SHORT'] and current_sl_corretora <= mon["be_price"] * 1.001:
                    ja_no_be = True

            if (not mon["be_ativado"] and not ja_no_be) and pnl_em_r >= 1.0:
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
        
        # O Saldo Estimado agora sobe ou desce permanentemente com o PnL Realizado Acumulado
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
                await self.executar_novas_entradas()
                await self.sincronizar_posicoes_abertas()
                await self.gerenciar_posicoes_2r()
                await self.relatorio_periodico()
            except Exception as e:
                logging.error(f"Erro crítico no loop do gerenciador: {e}")
            finally:
                await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador de risco 2:1 desligado pelo operador.")
