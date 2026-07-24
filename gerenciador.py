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
        self.last_summary_time = time.time()  
        self.banca_acumulada_real = float(Config.BANCA_DEMO_INICIAL) # 💰 Controlo real e persistente do capital

    def calcular_topologia_trade_blindada(self, signal: dict) -> dict:
        """
        🧠 INTELIGÊNCIA QUANTITATIVA COM ESCUDO DE RISCO:
        Calcula alavancagem e lote dinâmicos, mas impõe um Hard Cap institucional 
        (Máximo de 35x de alavancagem e teto de margem) para proteger a banca real.
        """
        score = float(signal.get('score', 5.0))
        prob = float(signal.get('prob', 50.0))
        current_atr = float(signal.get('current_atr', 0.01))
        price = float(signal.get('price', 1.0))

        # 1. Índice Sintético de Convicção (0.0 a 1.0)
        score_norm = np.clip(score / 10.0, 0.0, 1.0)
        prob_norm = np.clip(prob / 100.0, 0.0, 1.0)
        conviction = (score_norm * 0.6) + (prob_norm * 0.4)

        # 2. Alavancagem Dinâmica Blindada (Teto Máximo de 35x para timeframe de 15m)
        atr_pct = (current_atr / price) * 100 if price > 0 else 1.0
        fator_volatilidade = np.clip(1.0 / (atr_pct + 0.1), 0.2, 1.5)
        
        leverage_raw = 1.0 + (conviction ** 2) * 34.0 * fator_volatilidade
        leverage = int(np.clip(round(leverage_raw), 1, 35)) # 🛡️ HARD CAP: Nunca passa de 35x

        # 3. Lote/Margem em USD Contínuo (Teto seguro de $5.00 por ordem para não estourar 10 trades)
        min_usd = 1.00
        max_usd = 5.00
        invest_amount = min_usd + (conviction ** 1.5) * (max_usd - min_usd)
        invest_amount = float(np.clip(invest_amount, min_usd, max_usd))

        # 4. Vertente de Classificação
        if conviction >= 0.85:
            vertente = "CONVIÇÃO INSTITUCIONAL EXTREMA (SNIPER)"
            atr_multi_tp = 2.5
            atr_multi_sl = 1.0
        elif conviction >= 0.65:
            vertente = "MOMENTUM DE ALTA FREQUÊNCIA"
            atr_multi_tp = 2.0
            atr_multi_sl = 1.1
        else:
            vertente = "ADAPTATIVO DE ESCALA TÁTICA"
            atr_multi_tp = 1.5
            atr_multi_sl = 1.2

        # 5. Cálculo Matemático de Alvos baseados no ATR
        distancia_tp = current_atr * atr_multi_tp
        distancia_sl = current_atr * atr_multi_sl

        max_sl_pct = 0.020  # Trava o stop em no máximo 2% de oscilação
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
            "vertente": vertente,
            "conviction": conviction,
            "leverage": leverage,
            "invest_amount": round(invest_amount, 4),
            "tp": float(tp_price),
            "sl": float(sl_price),
            "qty": float(qty)
        }

    async def processar_sinais_inbound(self):
        """Avalia os sinais e dispara ordens com topologia blindada."""
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

            topology = self.calcular_topologia_trade_blindada(signal)

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
                    f"Margem Alocada: ${topology['invest_amount']:.2f} | Alavancagem: {topology['leverage']}x\n"
                    f"Convicção da IA: {topology['conviction']*100:.1f}% (Score: {signal['score']:.1f} | Prob: {signal['prob']:.1f}%)\n"
                    f"Preço Entrada: ${signal['price']:.5f}\n"
                    f"🎯 Take Profit: ${topology['tp']:.5f} | 🛑 Stop Loss: ${topology['sl']:.5f}\n"
                )
                await TelegramLogger.send(msg)
                logging.info(f"Ordem blindada processada para {symbol} | Alavancagem: {topology['leverage']}x | Margem: ${topology['invest_amount']:.2f}")

    async def gerenciar_defesa_elastica_balde(self):
        """
        🛡️ CATRACA MÓVEL & GESTÃO DE RISCO DE BANCA REAL
        Monitora posições e calcula o saldo real acumulado com base no capital inicial.
        """
        positions = await self.executor.execution.get_current_positions(self.db)
        now = time.time()

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            self.last_summary_time = now
            return

        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        total_margin = sum((float(p.get('contracts', 0)) * float(p.get('entryPrice', 0))) / max(1, float(p.get('leverage', 35))) for p in positions)

        if total_margin <= 0: return

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        # 🛡️ HARD STOP GLOBAL SEGURO: Baseado no capital total e não na margem alocada
        banca_base = float(Config.BANCA_DEMO_INICIAL)
        patrimonio_atual = banca_base + total_net_pnl
        
        # Se o PnL flutuante negativo consumir mais de 40% da banca total, o balde defende
        limite_perda_banca = -banca_base * 0.40 
        fase_catraca = "INATIVA"
        
        if max_roe >= 0.60:
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "ASFIXIA (Fase 3)"
        elif max_roe >= 0.35:
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.20:
            stop_dinamico_usd = 0.02 * total_margin
            fase_catraca = "BREAK-EVEN (Fase 1)"
        else:
            stop_dinamico_usd = limite_perda_banca

        # ⏱️ RELATÓRIO PERIÓDICO DE 10 MINUTOS (Raio-X do Balde Blindado)
        if now - self.last_summary_time >= 600:
            self.last_summary_time = now
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"

            resumo_msg = (
                f"⏱️ <b>RAIO-X DO BALDE (10 min)</b> [{modo_texto}]\n\n"
                f"🔹 <b>Operações Ativas:</b> {len(positions)}/{Config.MAX_OPEN_TRADES}\n"
                f"🔹 <b>Margem Alocada:</b> ${total_margin:.2f}\n"
                f"🔹 <b>PnL Atual Flutuante:</b> ${total_net_pnl:+.2f} ({current_roe * 100:.1f}% ROE)\n\n"
                f"📈 <b>Topo (Max PnL):</b> ${self.max_basket_pnl:.2f}\n"
                f"📉 <b>Fundo (Min PnL):</b> ${self.min_basket_pnl:.2f}\n\n"
                f"🔒 <b>Catraca Ativa:</b> {fase_catraca}\n"
                f"💰 <b>Patrimônio Estimado:</b> ${patrimonio_atual:.2f}\n"
                f"🛑 <b>Gatilho de Defesa Global:</b> ${stop_dinamico_usd:.2f}"
            )
            await TelegramLogger.send(resumo_msg)
            logging.info("⏱️ Relatório de Raio-X do Balde Blindado enviado para o Telegram.")

        # Disparo da Defesa Global
        if total_net_pnl <= stop_dinamico_usd:
            logging.warning(f"🚨 DEFESA GLOBAL ACIONADA: Fechamento preventivo via {fase_catraca}")
            
            for p in positions:
                await self.executor.force_close_position(
                    symbol=p['symbol'], 
                    side=p['side'], 
                    amount=float(p.get('contracts', 0)), 
                    outcome='BASKET_CLOSE_EMERGENCY', 
                    pnl=total_net_pnl / len(positions)
                )

            patrimonio_final = banca_base + total_net_pnl
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            msg_fechamento = (
                f"🚨 [{modo_texto}] DEFESA GLOBAL ACIONADA (CESTA LIQUIDADA)\n\n"
                f"📊 RELATÓRIO FINANCEIRO DE ENCERRAMENTO:\n"
                f"• Status da Catraca: {fase_catraca}\n"
                f"• PnL Realizado do Ciclo: ${total_net_pnl:+.2f}\n"
                f"• Patrimônio Final da Banca: ${patrimonio_final:.2f}\n"
            )
            await TelegramLogger.send(msg_fechamento)
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

    async def loop_agente_autonomo(self):
        """Loop contínuo de autonomia financeira e telemetria blindada."""
        logging.info("🧠 AGENTE AUTÔNOMO BLINDADO (Teto de 35x e Lotes Controlados) ONLINE")
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
            
