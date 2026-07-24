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

    def calcular_topologia_trade_avancada(self, signal: dict) -> dict:
        score = float(signal.get('score', 5.0))
        prob = float(signal.get('prob', 50.0))
        current_atr = float(signal.get('current_atr', 0.01))
        price = float(signal.get('price', 1.0))

        score_norm = np.clip(score / 10.0, 0.0, 1.0)
        prob_norm = np.clip(prob / 100.0, 0.0, 1.0)
        conviction = (score_norm * 0.6) + (prob_norm * 0.4)

        atr_pct = (current_atr / price) * 100 if price > 0 else 1.0
        fator_volatilidade = np.clip(1.0 / (atr_pct + 0.1), 0.2, 2.0)
        
        leverage_raw = 1.0 + (conviction ** 2) * 99.0 * fator_volatilidade
        leverage = int(np.clip(round(leverage_raw), 1, 100))

        min_usd = 0.50
        max_usd = 10.00 
        
        kelly_fraction = 0.25 
        base_invest = min_usd + (conviction ** 1.5) * (max_usd - min_usd)
        invest_amount = float(np.clip(base_invest * kelly_fraction, min_usd, max_usd))

        asset_type = signal.get('asset_type', 'CRYPTO')
        
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

        for signal in signals:
            symbol = signal['symbol']
            if symbol in open_symbols: continue

            tempo_liberacao = await self.db.get_cooldown(symbol)
            if time.time() < tempo_liberacao: continue
            if len(open_symbols) >= Config.MAX_OPEN_TRADES: break

            topology = self.calcular_topologia_trade_avancada(signal)

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

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            self.last_summary_time = now
            return

        banca_inicial = getattr(Config, 'BANCA_INICIAL', 100.0)
        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        
        # EXTRATOR BLINDADO DE MARGEM REAL
        total_margin = 0.0
        for p in positions:
            real_margin = float(p.get('initialMargin', 0.0))
            
            if real_margin == 0.0:
                # Busca as métricas brutas
                qty = float(p.get('contracts', p.get('amount', p.get('size', 0.0))))
                price = float(p.get('entryPrice', p.get('price', 0.0)))
                notional = qty * price
                
                # Caçada à alavancagem real nos metadados da corretora
                lev = 1.0
                if p.get('leverage'):
                    lev = float(p['leverage'])
                elif 'info' in p and isinstance(p['info'], dict):
                    info = p['info']
                    if info.get('leverage'):
                        lev = float(info['leverage'])
                    elif info.get('positionInitialMargin'): # Padrão Binance
                        real_margin = float(info['positionInitialMargin'])
                    elif info.get('positionMargin'): # Padrão Bybit
                        real_margin = float(info['positionMargin'])
                        
                # Se não achou a margem pronta, calcula com a alavancagem encontrada
                if real_margin == 0.0 and notional > 0:
                    lev = max(1.0, float(lev))
                    real_margin = notional / lev
                    
            total_margin += real_margin

        # Fallback anti-quebra
        if total_margin <= 0: total_margin = 1.0

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        # AGORA O ROE É MEDIDO SOBRE O CAPITAL EXPOSTO, NÃO SOBRE A BANCA PARADA
        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        # Stop Global de 70% apenas sobre a MARGEM ALOCADA (Ex: se alocou $8, stop é -$5.60)
        stop_dinamico_usd = -total_margin * 0.70  
        fase_catraca = "INATIVA"
        
        # MATEMÁTICA DA CATRACA DINÂMICA (Sobre a margem alocada)
        if max_roe >= 0.50: # 50% de lucro sobre o investido
            stop_dinamico_usd = (max_roe - 0.20) * total_margin
            fase_catraca = "ASFIXIA CONTÍNUA (Fase 3)"
        elif max_roe >= 0.25: # 25% de lucro sobre o investido
            stop_dinamico_usd = total_margin * 0.10
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.10: # 10% de lucro sobre o investido
            stop_dinamico_usd = 0.0
            fase_catraca = "BREAK-EVEN (Fase 1)"

        if now - self.last_summary_time >= 600:
            self.last_summary_time = now
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            saldo_atual = banca_inicial + total_net_pnl

            resumo_msg = (
                f"⏱️ <b>RAIO-X DO BALDE (10 min)</b> [{modo_texto}]\n\n"
                f"🔹 <b>Operações Ativas:</b> {len(positions)}/{Config.MAX_OPEN_TRADES}\n"
                f"🔹 <b>Margem Alocada Real:</b> ${total_margin:.2f}\n"
                f"🔹 <b>PnL Atual Flutuante:</b> ${total_net_pnl:+.2f} ({current_roe * 100:.1f}% sobre a Margem)\n\n"
                f"📈 <b>Topo (Max PnL):</b> ${self.max_basket_pnl:.2f} ({max_roe * 100:.1f}% ROE)\n"
                f"📉 <b>Fundo (Min PnL):</b> ${self.min_basket_pnl:.2f}\n\n"
                f"🔒 <b>Catraca Ativa:</b> {fase_catraca}\n"
                f"💰 <b>Saldo Estimado da Banca:</b> ${saldo_atual:.2f}\n"
                f"🛑 <b>Gatilho de Fechamento em:</b> ${stop_dinamico_usd:.2f}"
            )
            await TelegramLogger.send(resumo_msg)
            logging.info("⏱️ Relatório de Raio-X do Balde enviado com sucesso para o Telegram.")

        if total_net_pnl <= stop_dinamico_usd:
            logging.warning(f"🚨 DEFESA GLOBAL ACIONADA: Fechamento preventivo via {fase_catraca}")
            
            close_tasks = [
                self.executor.force_close_position(
                    symbol=p['symbol'], 
                    side=p['side'], 
                    amount=float(p.get('contracts', p.get('amount', 0))), 
                    outcome='BASKET_CLOSE_EMERGENCY', 
                    pnl=total_net_pnl / len(positions)
                ) for p in positions
            ]
            await asyncio.gather(*close_tasks, return_exceptions=True)

            saldo_atual_banca = banca_inicial + total_net_pnl
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            msg_fechamento = (
                f"🚨 [{modo_texto}] DEFESA GLOBAL ACIONADA (CESTA LIQUIDADA)\n\n"
                f"📊 RELATÓRIO FINANCEIRO DE ENCERRAMENTO:\n"
                f"• Status da Catraca: {fase_catraca}\n"
                f"• PnL Realizado do Ciclo: ${total_net_pnl:+.2f}\n"
                f"• Saldo Atual da Banca: ${saldo_atual_banca:.2f}\n"
            )
            await TelegramLogger.send(msg_fechamento)
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

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
                
