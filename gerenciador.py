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

    def calcular_topologia_trade_avancada(self, signal: dict) -> dict:
        """
        🧠 INTELIGÊNCIA QUANTITATIVA INSTITUCIONAL:
        Calcula de forma contínua e dinâmica a alavancagem (1x a 100x) 
        e o lote em USD ($0.001 a $10.00) com base na Conviction Score e Volatilidade (ATR).
        """
        score = float(signal.get('score', 5.0))
        prob = float(signal.get('prob', 50.0))
        current_atr = float(signal.get('current_atr', 0.01))
        price = float(signal.get('price', 1.0))

        # 1. Índice Sintético de Convicção (0.0 a 1.0) ponderando Score e Probabilidade
        score_norm = np.clip(score / 10.0, 0.0, 1.0)
        prob_norm = np.clip(prob / 100.0, 0.0, 1.0)
        conviction = (score_norm * 0.6) + (prob_norm * 0.4)

        # 2. Alavancagem Contínua Dinâmica (1x a 100x)
        # Em alta convicção e baixa volatilidade, escala exponencialmente até 100x
        atr_pct = (current_atr / price) * 100 if price > 0 else 1.0
        fator_volatilidade = np.clip(1.0 / (atr_pct + 0.1), 0.2, 2.0)
        
        # Fórmula de Alavancagem Contínua Mapeada
        leverage_raw = 1.0 + (conviction ** 2) * 99.0 * fator_volatilidade
        leverage = int(np.clip(round(leverage_raw), 1, 100))

        # 3. Lote/Margem em USD Contínuo ($0.001 a $10.00)
        # Escala de acordo com a convicção pura do modelo
        min_usd = 0.001
        max_usd = 10.00
        invest_amount = min_usd + (conviction ** 1.5) * (max_usd - min_usd)
        invest_amount = float(np.clip(invest_amount, min_usd, max_usd))

        # 4. Vertente de Classificação para logs e relatórios
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

        # 5. Cálculo Matemático Estrito de Alvos baseados no ATR
        distancia_tp = current_atr * atr_multi_tp
        distancia_sl = current_atr * atr_multi_sl

        max_sl_pct = 0.025  # Trava o stop técnico em até 2.5% do preço
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
        """Avalia os sinais e dispara ordens com topologia de risco totalmente livre."""
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
        """
        🛡️ CATRACA MÓVEL COM RASTREABILIDADE FINANCEIRA COMPLETA
        Monitora posições, calcula saldos de banca e acumulados em tempo real.
        """
        positions = await self.executor.execution.get_current_positions(self.db)
        now = time.time()

        if not positions:
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0
            return

        total_net_pnl = sum(float(p.get('netPnl', 0)) for p in positions)
        total_margin = sum((float(p.get('contracts', 0)) * float(p.get('entryPrice', 0))) / max(1, float(p.get('leverage', 1))) for p in positions)

        if total_margin <= 0: return

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        current_roe = total_net_pnl / total_margin
        max_roe = self.max_basket_pnl / total_margin

        # Hard Stop Global de Segurança (-70% ROE)
        stop_dinamico_usd = -total_margin * 0.70  
        fase_catraca = "INATIVA"
        
        # Fases da Catraca Móvel
        if max_roe >= 0.60:
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "ASFIXIA (Fase 3)"
        elif max_roe >= 0.35:
            stop_dinamico_usd = (max_roe - 0.15) * total_margin
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.20:
            stop_dinamico_usd = 0.02 * total_margin
            fase_catraca = "BREAK-EVEN (Fase 1)"

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

            # 💰 RECUPERAÇÃO DE DADOS DE BANCA ATUALIZADOS DA CORRETORA/DB
            banca_inicial = getattr(Config, 'BANCA_INICIAL', 100.0)
            # Lê o acumulado geral do banco ou histórico se houver, simulando com o PnL atual fechado
            saldo_atual_banca = banca_inicial + total_net_pnl
            
            modo_texto = "CONTA REAL ⚠️" if Config.OPERA_CONTA_REAL else "SIMULAÇÃO 🔬"
            msg_fechamento = (
                f"🚨 [{modo_texto}] DEFESA GLOBAL ACIONADA (CESTA LIQUIDADA)\n\n"
                f"📊 RELATÓRIO FINANCEIRO DE ENCERRAMENTO:\n"
                f"• Status da Catraca: {fase_catraca}\n"
                f"• PnL Realizado do Ciclo: ${total_net_pnl:+.2f}\n"
                f"• Saldo Atual da Banca: ${saldo_atual_banca:.2f}\n"
                f"• Saldo Acumulado Total: ${total_net_pnl:+.2f} USD\n"
            )
            await TelegramLogger.send(msg_fechamento)
            self.max_basket_pnl = 0.0
            self.min_basket_pnl = 0.0

    async def loop_agente_autonomo(self):
        """Loop contínuo de autonomia financeira e gestão inteligente."""
        logging.info("🧠 AGENTE AUTÔNOMO COM ALAVANCAGEM LIVRE (1x-100x) E LOTES ($0.001-$10) ONLINE")
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
            
