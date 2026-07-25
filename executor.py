import sys
import os
import asyncio
import logging
import time
import sqlite3
import pandas as pd
import numpy as np
import quantstats as qs
from config import Config
from executor import EngineExecutor, TelegramLogger

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [GERENCIADOR] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class GerenciadorRiscoAutonomo:
    def __init__(self):
        self.executor = EngineExecutor()
        self.db = self.executor.db
        self.max_basket_pnl = 0.0
        self.min_basket_pnl = 0.0
        self.last_summary_time = time.time()
        self.trade_history_buffer = []
        
        self.margens_ativas = {}
        self.pnl_realizado_acumulado = 0.0

    def calcular_topologia_trade_avancada(self, signal: dict, saldo_banca: float) -> dict:
        score = float(signal.get('score', 5.0))
        prob = float(signal.get('prob', 50.0))
        current_atr = float(signal.get('current_atr', 0.01))
        price = float(signal.get('price', 1.0))
        asset_type = signal.get('asset_type', 'CRYPTO')

        score_norm = np.clip(score / 10.0, 0.0, 1.0)
        prob_norm = np.clip(prob / 100.0, 0.0, 1.0)
        conviction = (score_norm * 0.6) + (prob_norm * 0.4)

        atr_pct = (current_atr / price) * 100 if price > 0 else 1.0
        fator_volatilidade = np.clip(1.0 / (atr_pct + 0.1), 0.2, 2.0)
        
        leverage_raw = 1.0 + (conviction ** 2) * 99.0 * fator_volatilidade
        
        # VERTENTE 1: Teto de Alavancagem por Mercado
        if asset_type == 'CRYPTO':
            leverage = int(np.clip(round(leverage_raw), 1, 50)) # Travado em 50x para parar de estourar operações
        elif asset_type == 'FOREX':
            leverage = int(np.clip(round(leverage_raw), 1, 100))
        else:
            leverage = int(np.clip(round(leverage_raw), 1, 20))

        # VERTENTE 2: Gestão de Lote Proporcional ao Saldo (Anti-Superconfiança)
        min_usd = 0.50
        max_usd_absoluto = 10.00
        
        limite_proporcional = saldo_banca * 0.05 # Nunca aloca mais de 5% da banca em uma única ordem
        max_usd_permitido = min(max_usd_absoluto, limite_proporcional)
        
        kelly_fraction = 0.25 
        base_invest = min_usd + (conviction ** 1.5) * (max_usd_permitido - min_usd)
        invest_amount = float(np.clip(base_invest * kelly_fraction, min_usd, max_usd_permitido))
        
        if asset_type == 'FOREX':
            atr_multi_tp, atr_multi_sl = 2.2, 0.9
        elif asset_type == 'STOCK':
            atr_multi_tp, atr_multi_sl = 2.5, 1.0
        else:
            if conviction >= 0.85:
                atr_multi_tp, atr_multi_sl = 3.0, 1.0
            elif conviction >= 0.65:
                atr_multi_tp, atr_multi_sl = 2.2, 1.0
            else:
                atr_multi_tp, atr_multi_sl = 1.8, 1.0

        distancia_tp = current_atr * atr_multi_tp
        distancia_sl = current_atr * atr_multi_sl

        max_sl_pct = 0.025  
        if (distancia_sl / price) > max_sl_pct:
            distancia_sl = price * max_sl_pct

        if signal['direction'] == 'BUY':
            tp_price = price + distancia_tp
            sl_price = price - distancia_sl
        else:
            tp_price = price - distancia_tp
            sl_price = price + distancia_sl

        qty = (invest_amount * leverage) / price if price > 0 else 0.0

        return {
            "vertente": f"VERTENTE_{asset_type}_{conviction:.2f}",
            "conviction": conviction,
            "leverage": leverage,
            "invest_amount": round(invest_amount, 4),
            "tp": float(tp_price),
            "sl": float(sl_price),
            "qty": float(qty)
        }

    async def processar_sinais_inbound(self):
        signals = await self.db.get_elite_signals()
        if not signals: return

        open_trades = await self.db.get_all_open_trades()
        open_symbols = {t[0] for t in open_trades}

        if len(open_symbols) >= Config.MAX_OPEN_TRADES: return

        banca_inicial = getattr(Config, 'BANCA_INICIAL', 100.0)
        saldo_atual_estimado = banca_inicial + self.pnl_realizado_acumulado

        for signal in signals:
            symbol = signal['symbol']
            if symbol in open_symbols: continue

            tempo_liberacao = await self.db.get_cooldown(symbol)
            if time.time() < tempo_liberacao: continue
            if len(open_symbols) >= Config.MAX_OPEN_TRADES: break

            # Passa o saldo estimado para calcular a alocação dinâmica
            topology = self.calcular_topologia_trade_avancada(signal, saldo_atual_estimado)

            order_packet = {
                'symbol': symbol,
                'direction': signal['direction'],
                'qty': topology['qty'],
                'current_price': signal['price'],
                'tp': topology['tp'],
                'sl': topology['sl']
            }

            sucesso = await self.executor.order_router_inbound(order_packet, signal)
            
            if sucesso:
                open_symbols.add(symbol)
                
                self.margens_ativas[symbol] = topology['invest_amount']
                
                modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
                msg = (
                    f"🔬 [{modo_texto}] AGENTE AUTÔNOMO DISPAROU | {topology['vertente']}\n\n"
                    f"Ativo: {symbol} | Direção: {signal['direction']}\n"
                    f"Margem Alocada: ${topology['invest_amount']:.4f} | Alavancagem: {topology['leverage']}x\n"
                    f"Convicção da IA: {topology['conviction']*100:.1f}% (Score: {signal['score']:.1f} | Prob: {signal['prob']:.1f}%)\n"
                    f"Preço Entrada: ${signal['price']:.5f}\n"
                    f"🎯 Take Profit: ${topology['tp']:.5f} | 🛑 Stop Loss: ${topology['sl']:.5f}\n"
                )
                await TelegramLogger.send(msg)
                logging.info(f"Ordem autônoma processada para {symbol} | Alavancagem: {topology['leverage']}x | Margem: ${topology['invest_amount']:.4f}")

    async def gerenciar_defesa_elastica_balde(self):
        try:
            positions = await self.executor.execution.get_current_positions(self.db)
        except Exception as e:
            logging.error(f"Erro resiliente ao buscar posições na corretora: {e}")
            return

        now = time.time()
        banca_inicial = getattr(Config, 'BANCA_INICIAL', 100.0)
        saldo_base_sessao = banca_inicial + self.pnl_realizado_acumulado

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            self.last_summary_time = now
            self.margens_ativas.clear() 
            return

        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        total_margin = 0.0
        symbols_ativos_corretora = set()
        
        # VERTENTE 3: Hard Stop Individual Anti-Sangramento
        operacoes_para_podar = []

        for p in positions:
            symbol = p.get('symbol')
            symbols_ativos_corretora.add(symbol)
            pnl_individual = float(p.get('netPnl', 0))
            
            if symbol in self.margens_ativas:
                real_margin = self.margens_ativas[symbol]
            else:
                qty = float(p.get('contracts', p.get('amount', p.get('size', 0.0))))
                price = float(p.get('entryPrice', p.get('price', 0.0)))
                notional = qty * price
                
                if getattr(Config, 'OPERA_CONTA_REAL', False):
                    lev = float(p.get('leverage', 1.0))
                    real_margin = notional / max(1.0, lev)
                else:
                    margem_estimada = notional / 50.0  
                    real_margin = float(np.clip(margem_estimada, 0.50, 10.00))
                    
                self.margens_ativas[symbol] = real_margin

            total_margin += real_margin
            
            # Checa se esta operação específica está afundando a cesta (> 80% de perda da margem dela)
            roe_individual = pnl_individual / real_margin if real_margin > 0 else 0
            if roe_individual <= -0.80:
                operacoes_para_podar.append((symbol, p.get('side'), float(p.get('contracts', p.get('amount', 0))), pnl_individual))

        self.margens_ativas = {sym: margem for sym, margem in self.margens_ativas.items() if sym in symbols_ativos_corretora}

        # Executa a poda das operações tóxicas isoladamente
        for symbol, side, amount, pnl_ind in operacoes_para_podar:
            logging.warning(f"🪓 PODA ATIVA: Cortando {symbol} por sangramento extremo (ROE < -80%).")
            await self.executor.force_close_position(symbol, side, amount, 'STOP_INDIVIDUAL_SANGRAMENTO', pnl_ind)
            self.pnl_realizado_acumulado += pnl_ind
            del self.margens_ativas[symbol]
            total_margin -= self.margens_ativas.get(symbol, 0)
            total_net_pnl -= pnl_ind

        # Se podou todas as operações, encerra a verificação desta rodada
        if total_margin <= 0: 
            return

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        stop_dinamico_usd = -total_margin * 0.70  
        fase_catraca = "INATIVA"
        
        if max_roe >= 0.70:
            stop_dinamico_usd = (max_roe - 0.20) * total_margin
            fase_catraca = "ASFIXIA CONTÍNUA (Fase 3)"
        elif max_roe >= 0.45:
            stop_dinamico_usd = total_margin * 0.15
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.25:
            stop_dinamico_usd = total_margin * 0.02
            fase_catraca = "BREAK-EVEN SEGURO (Fase 1)"

        saldo_atual_estimado = saldo_base_sessao + total_net_pnl

        if now - self.last_summary_time >= 600:
            self.last_summary_time = now
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"

            resumo_msg = (
                f"⏱️ <b>RAIO-X DO BALDE (10 min)</b> [{modo_texto}]\n\n"
                f"🔹 <b>Operações Ativas:</b> {len(positions) - len(operacoes_para_podar)}/{Config.MAX_OPEN_TRADES}\n"
                f"🔹 <b>Margem Alocada Real:</b> ${total_margin:.2f}\n"
                f"🔹 <b>PnL Atual Flutuante:</b> ${total_net_pnl:+.2f} ({current_roe * 100:.1f}% sobre a Margem)\n\n"
                f"📈 <b>Topo (Max PnL):</b> ${self.max_basket_pnl:.2f} ({max_roe * 100:.1f}% ROE)\n"
                f"📉 <b>Fundo (Min PnL):</b> ${self.min_basket_pnl:.2f}\n\n"
                f"🔒 <b>Catraca Ativa:</b> {fase_catraca}\n"
                f"💰 <b>Saldo Estimado da Banca:</b> ${saldo_atual_estimado:.2f}\n"
                f"🛑 <b>Gatilho de Fechamento em:</b> ${stop_dinamico_usd:.2f}"
            )
            await TelegramLogger.send(resumo_msg)
            logging.info("⏱️ Relatório de Raio-X do Balde enviado com sucesso para o Telegram.")

        if total_net_pnl <= stop_dinamico_usd:
            logging.warning(f"🚨 DEFESA GLOBAL ACIONADA: Fechamento preventivo via {fase_catraca}")
            
            # Buscar as posições novamente caso a "poda" tenha alterado a lista ativa
            posicoes_finais = await self.executor.execution.get_current_positions(self.db)
            
            close_tasks = [
                self.executor.force_close_position(
                    symbol=p['symbol'], 
                    side=p['side'], 
                    amount=float(p.get('contracts', p.get('amount', 0))), 
                    outcome='BASKET_CLOSE_EMERGENCY', 
                    pnl=float(p.get('netPnl', 0))
                ) for p in posicoes_finais
            ]
            await asyncio.gather(*close_tasks, return_exceptions=True)

            self.pnl_realizado_acumulado += total_net_pnl
            saldo_final_ciclo = banca_inicial + self.pnl_realizado_acumulado
            
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            msg_fechamento = (
                f"🚨 [{modo_texto}] DEFESA GLOBAL ACIONADA (CESTA LIQUIDADA)\n\n"
                f"📊 RELATÓRIO FINANCEIRO DE ENCERRAMENTO:\n"
                f"• Status da Catraca: {fase_catraca}\n"
                f"• PnL Realizado do Ciclo: ${total_net_pnl:+.2f}\n"
                f"• Saldo Atual da Banca: ${saldo_final_ciclo:.2f}\n"
            )
            await TelegramLogger.send(msg_fechamento)
            
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            self.margens_ativas.clear()

    async def loop_agente_autonomo(self):
        logging.info("🧠 AGENTE AUTÔNOMO COM TELEMETRIA PERIÓDICA (10m) ONLINE")
        while True:
            try:
                await self.processar_sinais_inbound()
                await self.gerenciar_defesa_elastica_balde()
            except Exception as e:
                logging.error(f"Erro crítico no loop do gerenciador: {e}")
            finally:
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador autônomo desligado pelo operador.")
                
