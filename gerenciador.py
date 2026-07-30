import sys
import os
import asyncio
import logging
import time
import aiosqlite
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

        # Estado interno por símbolo (Bot Direcional)
        self.posicoes_monitoradas = {}
        self.fechamento_em_andamento = {} 

        # Estado interno por par (Bot Arbitragem)
        self.pares_arbitragem_monitorados = {}
        self.pernas_arbitragem_ativas = set() # Set para bloquear a interferência do bot direcional

        self.last_summary_time = time.time()
        self.pnl_realizado_acumulado = 0.0

    @property
    def saldo_atual(self) -> float:
        """Calcula o saldo da banca em tempo real"""
        banca_inicial = getattr(Config, 'BANCA_DEMO_INICIAL', 100.0)
        return banca_inicial + self.pnl_realizado_acumulado

    async def setup_db_arbitragem(self):
        """Garante que a tabela de sinais de arbitragem exista para não quebrar a VPS"""
        db_name = getattr(Config, 'DB_NAME', 'predador_v31.db')
        async with aiosqlite.connect(db_name) as db_conn:
            await db_conn.execute('''
                CREATE TABLE IF NOT EXISTS sinais_arbitragem (
                    par_id TEXT PRIMARY KEY,
                    leg_long TEXT,
                    leg_short TEXT,
                    price_long REAL,
                    price_short REAL,
                    target_pnl_usd REAL,
                    stop_pnl_usd REAL
                )
            ''')
            await db_conn.commit()

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

    # =========================================================================
    # MOTOR 1: GESTÃO DIRECIONAL (Mantido Intacto)
    # =========================================================================
    async def executar_novas_entradas(self):
        try:
            sinais = await self.db.get_elite_signals()
            if not sinais:
                return

            positions = await self.executor.execution.get_current_positions(self.db)
            max_trades = getattr(Config, 'MAX_OPEN_TRADES', 5)
            
            # Filtra apenas os símbolos que NÃO são de arbitragem
            simbolos_abertos = [
                p.get('symbol') for p in positions 
                if p.get('symbol') not in self.pernas_arbitragem_ativas
            ]

            if len(simbolos_abertos) >= max_trades:
                return

            for sinal in sinais:
                if len(simbolos_abertos) >= max_trades:
                    break

                symbol = sinal['symbol']
                direction = sinal['direction']
                entry = sinal['price']

                # Bloqueio duplo: Se a moeda já está na arbitragem, não opera ela aqui
                if symbol in simbolos_abertos or symbol in self.pernas_arbitragem_ativas:
                    continue

                cooldown = await self.db.get_cooldown(symbol)
                if time.time() < cooldown:
                    continue

                distancia_sl_pct = 0.005 
                
                if direction in ['BUY', 'LONG']:
                    sl = entry * (1 - distancia_sl_pct)
                else:
                    sl = entry * (1 + distancia_sl_pct)

                alvos = self._calcular_risco_e_alvos(entry, sl, direction)
                tp = alvos['tp']

                banca_neste_momento = self.saldo_atual
                risco_pct = getattr(Config, 'MAX_POSITION_RISK', 0.01)
                risco_usd = banca_neste_momento * risco_pct

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

                logging.info(f"⚡ ORDEM DIRECIONAL: {symbol} | {direction} | Qty: {qty:.4f} | Risco: ${risco_usd:.2f}")
                sucesso = await self.executor.order_router_inbound(order_packet)

                if sucesso:
                    msg = (
                        f"🎯 <b>NOVA POSIÇÃO DIRECIONAL ABERTA</b>\n"
                        f"Moeda: {symbol}\n"
                        f"Direção: {direction}\n"
                        f"Entrada: {entry:.5f}\n"
                        f"Take Profit (2R): {tp:.5f}\n"
                        f"Score IA: {sinal.get('score', 0):.1f}/10\n"
                        f"💰 Saldo da Banca: ${banca_neste_momento:.2f}"
                    )
                    await TelegramLogger.send(msg)
                    simbolos_abertos.append(symbol)

            await self.db.clear_elite_signals()

        except Exception as e:
            logging.error(f"Erro no módulo de execução direcional: {e}")

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
            
            # IGNORA COMPLETAMENTE AS MOEDAS QUE ESTÃO NA ARBITRAGEM
            if symbol in self.pernas_arbitragem_ativas:
                continue

            if symbol in self.fechamento_em_andamento:
                if time.time() - self.fechamento_em_andamento[symbol] < 30:
                    continue 
                else:
                    del self.fechamento_em_andamento[symbol]

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
                        current_sl = entry * 0.995
                    else:
                        current_sl = entry * 1.005

                alvos = self._calcular_risco_e_alvos(entry, current_sl, side)

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
                except Exception as e:
                    pass

            else:
                mon = self.posicoes_monitoradas[symbol]
                mon["last_net_pnl"] = net_pnl
                if net_pnl > mon["max_pnl"]:
                    mon["max_pnl"] = net_pnl
                if net_pnl < mon["min_pnl"]:
                    mon["min_pnl"] = net_pnl

        symbols_para_remover = [s for s in self.posicoes_monitoradas if s not in symbols_ativos]
        for s in symbols_para_remover:
            pnl_fechamento = self.posicoes_monitoradas[s].get("last_net_pnl", 0.0)
            self.pnl_realizado_acumulado += pnl_fechamento
            
            msg = (
                f"🏁 <b>POSIÇÃO DIRECIONAL ENCERRADA</b>\n"
                f"Moeda: {s}\n"
                f"PnL Realizado: ${pnl_fechamento:+.2f}\n"
                f"💰 Novo Saldo da Banca: ${self.saldo_atual:.2f}"
            )
            await TelegramLogger.send(msg)
            del self.posicoes_monitoradas[s]

    async def gerenciar_posicoes_2r(self):
        if not self.posicoes_monitoradas:
            return

        try:
            positions = await self.executor.execution.get_current_positions(self.db)
        except Exception:
            return

        pos_dict = {p.get('symbol'): p for p in positions if p.get('symbol') and p.get('symbol') not in self.pernas_arbitragem_ativas}
        db_name = getattr(Config, 'DB_NAME', 'predador_v31.db')

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

            if pnl_em_r >= 1.98 or pnl_em_r <= -1.0:
                self.fechamento_em_andamento[symbol] = time.time()
                self.pnl_realizado_acumulado += net_pnl
                
                tipo_msg = "🏆 ALVO 2R ATINGIDO" if pnl_em_r >= 1.98 else "🚨 STOP LOSS EXECUTADO"
                msg = (
                    f"{tipo_msg}\n"
                    f"Moeda: {symbol}\n"
                    f"Resultado: ${net_pnl:+.2f} ({pnl_em_r:.2f}R)\n"
                    f"💰 Novo Saldo da Banca: ${self.saldo_atual:.2f}"
                )
                await TelegramLogger.send(msg)
                
                try:
                    close_side = "SELL" if side in ["BUY", "LONG"] else "BUY"
                    if getattr(Config, 'OPERA_CONTA_REAL', False):
                        await self.executor.execution.exchange.create_order(
                            symbol=symbol, type='market', side=close_side,
                            amount=qty, params={'reduceOnly': True}
                        )
                except Exception as e:
                    logging.error(f"[{symbol}] Falha ao fechar a mercado: {e}")
                finally:
                    async with aiosqlite.connect(db_name, timeout=30) as db_conn:
                        await db_conn.execute("DELETE FROM trades WHERE symbol = ?", (symbol,))
                        await db_conn.commit()
                    
                    if symbol in self.posicoes_monitoradas:
                        del self.posicoes_monitoradas[symbol]
                continue 

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
                        symbol=symbol, take_profit=mon["current_tp"], stop_loss=novo_sl
                    )
                    mon["current_sl"] = novo_sl
                    mon["be_ativado"] = True
                except Exception:
                    pass

    # =========================================================================
    # MOTOR 2: GESTÃO DE ARBITRAGEM ESTATÍSTICA (NOVO)
    # =========================================================================
    async def executar_novas_entradas_arbitragem(self):
        try:
            db_name = getattr(Config, 'DB_NAME', 'predador_v31.db')
            async with aiosqlite.connect(db_name) as db_conn:
                db_conn.row_factory = aiosqlite.Row
                cursor = await db_conn.execute("SELECT * FROM sinais_arbitragem")
                sinais_arb = await cursor.fetchall()

            if not sinais_arb:
                return

            max_pares = getattr(Config, 'ARB_MAX_PARES_ATIVOS', 5)

            for sinal in sinais_arb:
                if len(self.pares_arbitragem_monitorados) >= max_pares:
                    break

                par_id = sinal['par_id']
                leg_long = sinal['leg_long']
                leg_short = sinal['leg_short']

                if par_id in self.pares_arbitragem_monitorados:
                    continue

                # Evita operar moedas que o direcional já está usando
                if leg_long in self.posicoes_monitoradas or leg_short in self.posicoes_monitoradas:
                    continue

                # Capital Market Neutral: O mesmo valor em USD em ambas as pontas
                banca = self.saldo_atual
                alocacao_usd_por_perna = banca * getattr(Config, 'ARB_MAX_POSITION_RISK', 0.01)

                qty_long = alocacao_usd_por_perna / sinal['price_long']
                qty_short = alocacao_usd_por_perna / sinal['price_short']

                # Pacote Long
                sucesso_long = await self.executor.order_router_inbound({
                    'symbol': leg_long, 'direction': 'BUY',
                    'qty': qty_long, 'current_price': sinal['price_long'],
                    'tp': 0.0, 'sl': 0.0 # O controle será via software pelo PnL Agregado
                })

                # Pacote Short
                sucesso_short = await self.executor.order_router_inbound({
                    'symbol': leg_short, 'direction': 'SELL',
                    'qty': qty_short, 'current_price': sinal['price_short'],
                    'tp': 0.0, 'sl': 0.0 
                })

                if sucesso_long and sucesso_short:
                    self.pares_arbitragem_monitorados[par_id] = {
                        "leg_long": leg_long, "leg_short": leg_short,
                        "qty_long": qty_long, "qty_short": qty_short,
                        "target_pnl": sinal['target_pnl_usd'],
                        "stop_pnl": sinal['stop_pnl_usd']
                    }
                    self.pernas_arbitragem_ativas.add(leg_long)
                    self.pernas_arbitragem_ativas.add(leg_short)

                    msg = (
                        f"⚖️ <b>NOVO PAR DE ARBITRAGEM ABERTO</b>\n"
                        f"Par: {par_id}\n"
                        f"Long: {leg_long} (${alocacao_usd_por_perna:.2f})\n"
                        f"Short: {leg_short} (${alocacao_usd_por_perna:.2f})\n"
                        f"Target Sintético: ${sinal['target_pnl_usd']:.2f}\n"
                        f"Stop Sintético: ${sinal['stop_pnl_usd']:.2f}"
                    )
                    await TelegramLogger.send(msg)

                    # Limpa o sinal executado
                    async with aiosqlite.connect(db_name) as db_conn:
                        await db_conn.execute("DELETE FROM sinais_arbitragem WHERE par_id = ?", (par_id,))
                        await db_conn.commit()

        except Exception as e:
            logging.error(f"Erro na abertura de arbitragem: {e}")

    async def gerenciar_posicoes_arbitragem(self):
        if not self.pares_arbitragem_monitorados:
            return

        try:
            positions = await self.executor.execution.get_current_positions(self.db)
            pos_dict = {p.get('symbol'): p for p in positions}
        except Exception:
            return

        pares_para_fechar = []

        for par_id, dados in self.pares_arbitragem_monitorados.items():
            l_sym, s_sym = dados['leg_long'], dados['leg_short']
            
            pnl_long = float(pos_dict.get(l_sym, {}).get('netPnl', 0.0)) if l_sym in pos_dict else 0.0
            pnl_short = float(pos_dict.get(s_sym, {}).get('netPnl', 0.0)) if s_sym in pos_dict else 0.0
            
            pnl_sintetico_total = pnl_long + pnl_short

            atingiu_alvo = pnl_sintetico_total >= dados['target_pnl']
            atingiu_stop = pnl_sintetico_total <= dados['stop_pnl']

            if atingiu_alvo or atingiu_stop:
                pares_para_fechar.append((par_id, l_sym, s_sym, pnl_sintetico_total, atingiu_alvo))

        # Fechamento Simultâneo
        for par_id, l_sym, s_sym, pnl_total, win in pares_para_fechar:
            self.pnl_realizado_acumulado += pnl_total
            icone = "🎯 WIN SINTÉTICO" if win else "⚠️ STOP SINTÉTICO"
            
            try:
                if getattr(Config, 'OPERA_CONTA_REAL', False):
                    # Fecha Leg Long
                    await self.executor.execution.exchange.create_order(
                        symbol=l_sym, type='market', side='SELL',
                        amount=self.pares_arbitragem_monitorados[par_id]['qty_long'], params={'reduceOnly': True}
                    )
                    # Fecha Leg Short
                    await self.executor.execution.exchange.create_order(
                        symbol=s_sym, type='market', side='BUY',
                        amount=self.pares_arbitragem_monitorados[par_id]['qty_short'], params={'reduceOnly': True}
                    )
            except Exception as e:
                logging.error(f"[{par_id}] Erro ao forçar fechamento duplo da arbitragem: {e}")
            finally:
                msg = (
                    f"{icone} <b>(ARBITRAGEM)</b>\n"
                    f"Par: {par_id}\n"
                    f"Resultado do Spread: ${pnl_total:+.2f}\n"
                    f"💰 Novo Saldo da Banca: ${self.saldo_atual:.2f}"
                )
                await TelegramLogger.send(msg)
                
                # Libera a memória para novos trades
                self.pernas_arbitragem_ativas.discard(l_sym)
                self.pernas_arbitragem_ativas.discard(s_sym)
                del self.pares_arbitragem_monitorados[par_id]

    async def relatorio_periodico(self):
        now = time.time()
        if now - self.last_summary_time < 600:
            return

        self.last_summary_time = now

        # Para não sobrecarregar, consolidaremos num relatório só.
        modo_texto = "CONTA REAL ⚠️" if getattr(Config, 'OPERA_CONTA_REAL', False) else "SIMULAÇÃO 🔬"
        await TelegramLogger.send(f"⏱️ <b>RAIO-X GERENCIAL (10 min)</b> [{modo_texto}]\n💰 Saldo Atual: ${self.saldo_atual:.2f}")

    async def loop_agente_autonomo(self):
        logging.info("🧠 GERENCIADOR HÍBRIDO ONLINE — Direcional & Arbitragem")
        logging.info(f"💰 Saldo da Banca Inicializado: ${self.saldo_atual:.2f}")
        
        await self.setup_db_arbitragem()
        
        while True:
            try:
                # Trilha Direcional
                await self.executar_novas_entradas()
                await self.sincronizar_posicoes_abertas()
                await self.gerenciar_posicoes_2r()

                # Trilha Arbitragem Neutra
                await self.executar_novas_entradas_arbitragem()
                await self.gerenciar_posicoes_arbitragem()

                await self.relatorio_periodico()
            except Exception as e:
                logging.error(f"Erro crítico no loop híbrido do gerenciador: {e}")
            finally:
                await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        gerenciador = GerenciadorRiscoAutonomo()
        asyncio.run(gerenciador.loop_agente_autonomo())
    except KeyboardInterrupt:
        logging.info("🛑 Gerenciador Híbrido desligado pelo operador.")
