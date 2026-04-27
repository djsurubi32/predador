import ccxt
import pandas as pd
import pandas_ta as ta
import feedparser
import schedule
import time
import json
import logging
import requests
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÃO DE ELITE
# ==============================================================================
class Config:
    # APIs
    BYBIT_API_KEY = "PCbYFGTo56hHRlttEH"
    BYBIT_SECRET = "4gwVzhfrjBWosLLxKxoXAtn8Bq78Zkuydw07"
    GEMINI_API_KEY = "AIzaSyCgrmRuBuJWXoCExve06Wk76T-UPryCozw"
    
    # Telegram Config
    TELEGRAM_TOKEN = "8525608206:AAFBVh0p8BBEfYlz9fSCSLCFKo1TyA7ubiM"
    TELEGRAM_CHAT_ID = "8383496275"
    
    OPERA_CONTA_REAL = False  
    VALOR_ENTRADA = 5.0        
    ALAVANCAGEM = 10           
    
    ATIVOS = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    NEWS_FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/"
    ]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# ==============================================================================
# COMUNICAÇÃO E MONITORAMENTO
# ==============================================================================
class TelegramLogger:
    @staticmethod
    def send(message):
        if not Config.TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        except: pass

# ==============================================================================
# IA E ANÁLISE DE MERCADO
# ==============================================================================
class GeminiCore:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

    def ask_decision(self, symbol, data_pack, news_summary):
        prompt = {
            "contents": [{
                "parts": [{
                    "text": f"Ativo: {symbol}. Dados Técnicos: {data_pack}. Notícias do Momento: {news_summary}. "
                            "Sua resposta deve ser um JSON purista, sem markdown: "
                            '{"action": "BUY", "reasoning": "Sua explicação técnica detalhada aqui", "stop_loss": 0.0, "take_profit": 0.0}'
                }]
            }]
        }
        try:
            response = requests.post(self.url, json=prompt, timeout=30)
            res_json = response.json()
            raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
            clean_text = raw_text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_text)
        except:
            return {"action": "HOLD", "reasoning": "Falha na comunicação com a IA."}

# ==============================================================================
# MOTOR PREDADOR COM MONITORAMENTO
# ==============================================================================
class PredatorBot:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': Config.BYBIT_API_KEY, 'secret': Config.BYBIT_SECRET,
            'enableRateLimit': True, 'options': {'defaultType': 'swap'}
        })
        self.ai = GeminiCore(Config.GEMINI_API_KEY)
        # Memória para monitorar operações Demo { symbol: {entry, sl, tp, side} }
        self.open_trades_demo = {}
        TelegramLogger.send("🚀 *Bot Predador Online*\nMonitoramento Ativo 24/7")

    def check_position_exists(self, symbol):
        """Verifica se já existe operação aberta para o ativo"""
        if Config.OPERA_CONTA_REAL:
            try:
                pos = self.exchange.fetch_position(symbol)
                return float(pos['contracts']) > 0
            except: return False
        return symbol in self.open_trades_demo

    def monitor_trades(self, symbol, current_price):
        """Acompanha o resultado da operação e avisa no desfecho"""
        if Config.OPERA_CONTA_REAL:
            # No real, a Bybit gerencia SL/TP, o bot apenas verifica se a posição sumiu
            return 

        if symbol in self.open_trades_demo:
            trade = self.open_trades_demo[symbol]
            outcome = None
            
            if trade['side'] == 'BUY':
                if current_price >= trade['tp']: outcome = "✅ TAKE PROFIT (LUCRO)"
                elif current_price <= trade['sl']: outcome = "❌ STOP LOSS (PREJUÍZO)"
            else:
                if current_price <= trade['tp']: outcome = "✅ TAKE PROFIT (LUCRO)"
                elif current_price >= trade['sl']: outcome = "❌ STOP LOSS (PREJUÍZO)"

            if outcome:
                msg = f"🏁 *DESFECHO DA OPERAÇÃO ({symbol})*\nResultado: {outcome}\nPreço Final: {current_price}"
                TelegramLogger.send(msg)
                del self.open_trades_demo[symbol]

    def run_cycle(self):
        logging.info("Iniciando ciclo...")
        
        # 1. Captura de Notícias RSS
        news_summary = ""
        for url in Config.NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:2]:
                    news_summary += f"• {entry.title}\n"
            except: pass

        for symbol in Config.ATIVOS:
            try:
                # 2. Dados de Mercado
                bars = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df.ta.rsi(append=True)
                df.ta.ema(length=20, append=True)
                current_price = df.iloc[-1]['close']
                last_data = df.iloc[-1].to_dict()

                # 3. Monitoramento e Bloqueio
                self.monitor_trades(symbol, current_price)
                
                if self.check_position_exists(symbol):
                    logging.info(f"[{symbol}] Operação em curso. Pulando análise.")
                    continue

                # 4. Transparência: Enviar o que será analisado
                analysis_msg = (f"🔍 *Analisando {symbol}*\n\n"
                               f"📈 *Preço:* {current_price}\n"
                               f"📰 *Notícias:* \n{news_summary}")
                TelegramLogger.send(analysis_msg)

                # 5. Decisão da IA
                decision = self.ai.ask_decision(symbol, last_data, news_summary)
                
                # Enviar Raciocínio da IA
                ia_msg = (f"🤖 *Raciocínio da IA ({symbol}):*\n"
                          f"Decisão: *{decision['action']}*\n"
                          f"Motivo: _{decision['reasoning']}_")
                TelegramLogger.send(ia_msg)

                if decision['action'] != "HOLD":
                    self.execute(symbol, decision, current_price)
                
                time.sleep(8) # Anti-bloqueio Google

            except Exception as e:
                logging.error(f"Erro no ativo {symbol}: {e}")
                time.sleep(8)

    def execute(self, symbol, decision, price):
        qty = round((Config.VALOR_ENTRADA * Config.ALAVANCAGEM) / price, 4)
        
        if not Config.OPERA_CONTA_REAL:
            self.open_trades_demo[symbol] = {
                'side': decision['action'], 'entry': price,
                'sl': decision['stop_loss'], 'tp': decision['take_profit']
            }
            report = (f"🧪 *OPERAÇÃO DEMO INICIADA*\nPar: {symbol}\nLado: {decision['action']}\n"
                      f"Entrada: {price}\nSL: {decision['stop_loss']} | TP: {decision['take_profit']}")
            TelegramLogger.send(report)
            return

        try:
            side = 'buy' if decision['action'] == "BUY" else 'sell'
            self.exchange.create_order(symbol, 'market', side, qty, params={
                'stopLoss': decision['stop_loss'], 'takeProfit': decision['take_profit']
            })
            TelegramLogger.send(f"✅ *ORDEM REAL EXECUTADA EM {symbol}*")
        except Exception as e:
            TelegramLogger.send(f"❌ *ERRO BYBIT:* {e}")

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    bot = PredatorBot()
    schedule.every(2).minutes.do(bot.run_cycle)
    bot.run_cycle() # Primeira rodada imediata
    
    while True:
        schedule.run_pending()
        time.sleep(1)