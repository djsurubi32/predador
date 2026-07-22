import sys
import os
import asyncio
import logging
import time
import sqlite3
import pandas as pd
import numpy as np
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

    def calcular_topologia_trade(self, signal: dict) -> dict:
        """
        🧠 INTELIGÊNCIA QUANTITATIVA: 
        Calcula dinamicamente Alavancagem, Lote, TP, SL e níveis de Catraca baseados no ATR e Score.
        """
        score = float(signal['score'])
        prob = float(signal['prob'])
        current_atr = float(signal['current_atr'])
        price = float(signal['price'])
        
        # 1. Alavancagem Dinâmica Inteligente (Baseada na Volatilidade e Score)
        atr_pct = (current_atr / price) * 100
        if atr_pct > 2.5:
            leverage = max(15, int(Config.ALAVANCAGEM * 0.5))
        elif score >= 9.0 and prob >= 75.0:
            leverage = int(Config.ALAVANCAGEM)
        else:
            leverage = int(Config.ALAVANCAGEM)

        # 2. Dimensionamento de Lote (Quantos USD investir na margem)
        if score >= 9.0 and prob >= 75.0:
            invest_amount = 6.0  # Lote Sniper Pesado
            vertente = "QUALIDADE EXTREMA (SNIPER)"
            atr_multi_tp = 2.5   
            atr_multi_sl = 1.2   
        elif "Fluxo:3.0" in signal['reasoning'].replace(" ", ""):
            invest_amount = 4.0  # Lote Padrão de Momentum
            vertente = "SCALPING DE MOMENTUM"
            atr_multi_tp = 1.8
            atr_multi_sl = 1.0
        else:
            invest_amount = 2.0  # Lote de Teste / Padrão Adaptativo
            vertente = "PADRÃO ADAPTATIVO"
            atr_multi_tp = 1.5
            atr_multi_sl = 1.0

        # 3. Cálculo Matemático dos Alvos e Stops Baseados na Volatilidade Real (ATR)
        distancia_tp = current_atr * atr_multi_tp
        distancia_sl = current_atr * atr_multi_sl

        max_sl_pct = 0.020  # Trava o stop em no máximo 2% de oscilação do preço
        if (distancia_sl / price) > max_sl_pct:
            distancia_sl = price * max_sl_pct

        if signal['direction'] == 'BUY':
            tp_price = price + distancia_tp
            sl_price = price - distancia_sl
        else:
            tp_price = price - distancia_tp
            sl_price = price + distancia_sl

        return {
            "vertente": vertente,
            "leverage": leverage,
            "invest_amount": invest_amount,
            "tp": float(tp_price),
            "sl": float(sl_price),
            "qty": float((invest_amount * leverage) / price)
        }

    async def processar_sinais_inbound(self):
        """Avalia a mesa do leilão e comanda o envio de ordens calculadas para o executor."""
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

            topology = self.calcular_topologia_trade(signal)

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
                    f"Margem: ${topology['invest_amount']:.2f} | Alavancagem: {topology['leverage']}x\n"
                    f"Preço Entrada: ${signal['price']:.5f}\n"
                    f"🎯 Take Profit Técnico: ${topology['tp']:.5f}\n"
                    f"🛑 Stop Loss Técnico: ${topology['sl']:.5f}\n"
                    f"Score do Radar: {signal['score']:.1f}/10.0 | Probabilidade: {signal['prob']:.1f}%\n"
                )
                await TelegramLogger.send(msg)
                logging.info(f"Ordem comandada com sucesso para {symbol} ({signal['direction']})")

    async def gerenciar_defesa_elastica_balde(self):
        """
        🛡️ CATRACA MÓVEL EM FASES (OTIMIZADA) E HARD STOP GLOBAL
        Monitora posições ativas dando espaço para o trade respirar.
        """
        positions = await self.executor.execution.get_current_positions(self.db)
        now = time.time()

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            return

        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        total_margin = sum((float(p.get('contracts', 0)) * float(p.get('entryPrice', 0))) / Config.ALAVANCAGEM for p in positions)

        if total_margin <= 0: return

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        # 🛑 HARD STOP GLOBAL: Proteção contra Cisne Negro (-70% ROE)
        stop_dinamico_usd = -total_margin * 0.70  
        fase_catraca = "INATIVA"
        
        # ⚡ SISTEMA DE ASFIXIA OTIMIZADO (Fases com folga para surf de tendência)
        if max_roe >= 0.60:
            # Fase 3 (Asfixia Final): Acima de 60% de ROE, trava recuando 15%
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "ASFIXIA (Fase 3)"
        elif max_roe >= 0.35:
            # Fase 2 (Fixação): Acima de 35% de ROE, trava garantindo o lucro substancial
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.20:
            # Fase 1 (Break-Even): Só ativa quando atinge 20% de ROE (evita fechar nos primeiros cêntimos)
            stop_dinamico_usd = 0.02 * total_margin
            fase_catraca = "BREAK-EVEN (Fase 1)"

        if total_net_pnl <= stop_dinamico_usd:
            logging.warning(f"🚨 COMANDO DE DEFESA ATIVADO: Fechamento global disparado. Motivo: {fase_catraca if fase_catraca != 'INATIVA' else 'Hard Stop Global'}")
            
            for p in positions:
                await self.executor.force_close_position(
                    symbol=p['symbol'], 
                    side=p['side'], 
                    amount=float(p.get('contracts', 0)), 
                    outcome='BASKET_CLOSE_EMERGENCY', 
                    pnl=total_net_pnl / len(positions)
                )

            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            msg_fechamento = (
                f"🚨 [{modo_texto}] DEFESA GLOBAL ACIONADA\n\n"
                f"A Cesta de operações foi liquidada preventivamente.\n"
                f"Status da Catraca: {fase_catraca}\n"
                f"PnL Agregado Realizado: ${total_net_pnl:.2f}\n"
            )
            await TelegramLogger.send(msg_fechamento)
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

    async def loop_agente_autonomo(self):
        """Loop infinito de monitoramento e decisões em tempo real."""
        logging.info("🧠 AGENTE AUTÔNOMO DE GESTÃO DE RISCO ONLINE (100% INDEPENDENTE)")
        while True:
            try:
                await self.processar_sinais_inbound()
                await self.gerenciar_defesa_elastica_balde()
            except Exception as e:
                logging.error(f"Erro no loop do gerenciador autônomo: {e}")
            finally:
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador de risco encerrado pelo usuário.")
