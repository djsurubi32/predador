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

    def calcular_topologia_trade_avancada(self, signal: dict) -> dict:
        """
        🧠 INTELIGÊNCIA QUANTITATIVA INSTITUCIONAL & CONTROLE DE MARGEM REAL:
        Garante que o capital alocado por trade respeite estritamente o teto da banca.
        """
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

        # Respeita o teto máximo de investimento por ordem (limite seguro para banca pequena)
        min_usd = 0.50
        max_usd = 10.00 # Teto individual por trade para não estourar a banca de $100-$133
        
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
        
        # Correção crucial: Margem real alocada dividida pela alavancagem de cada contrato
        total_margin = sum((float(p.get('contracts', 0)) * float(p.get('entryPrice', 0))) / max(1, float(p.get('leverage', 1))) for p in positions)
        
        # Limita visualmente e de forma lógica para não ultrapassar a banca real
        if total_margin > banca_inicial * 2:
            total_margin = sum(float(p.get('initialMargin', 5.0)) for p in positions) if 'initialMargin' in positions[0] else banca_inicial * 0.8

        if total_margin <= 0: total_margin = 1.0

        if total_net_pnl > self.max_basket_pnl: self.max_basket_pnl = total_net_pnl
        if total_net_pnl < self.min_basket_pnl: self.min_basket_pnl = total_net_pnl

        # Cálculo do ROE baseado na Banca Inicial de $100 (ou capital real da conta)
        current_roe = total_net_pnl / banca_inicial
        max_roe = self.max_basket_pnl / banca_inicial

        # Hard Stop Global de Segurança (-30% da banca total)
        stop_dinamico_usd = -banca_inicial * 0.30  
        fase_catraca = "INATIVA"
        
        # Fases da Catraca corrigidas para percentuais reais da banca ($100)
        if max_roe >= 0.25: # Com 25% de lucro na banca ($25), trava o lucro em 10%
            stop_dinamico_usd = banca_inicial * 0.10
            fase_catraca = "ASFIXIA (Fase 3)"
        elif max_roe >= 0.15: # Com 15% de lucro ($15), trava em 5%
            stop_dinamico_usd = banca_inicial * 0.05
            fase_catraca = "FIXAÇÃO (Fase 2)"
        elif max_roe >= 0.08: # Com 8% de lucro ($8), vai para o Break-Even (0%)
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
                f"🔹 <b>PnL Atual Flutuante:</b> ${total_net_pnl:+.2f} ({current_roe * 100:.1f}% sobre a Banca)\n\n"
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
                    amount=float(p.get('contracts', 0)), 
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
