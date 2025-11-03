import os
import asyncio
import logging
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from brain import chatbot_response, HELP_TEXT   # ✅ import HELP_TEXT from brain.py
from Brain_new import chatbot_response
# Typing indicator helper
import threading, requests, time

def start_typing(token, chat_id, stop_event):
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    payload = {"chat_id": chat_id, "action": "typing"}
    while not stop_event.is_set():
        try:
            requests.post(url, json=payload, timeout=3)
        except:
            pass
        stop_event.wait(2.5)

# Chat handler using Brain
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    # start typing indicator
    stop_evt = threading.Event()
    thr = threading.Thread(
        target=start_typing,
        args=(TOKEN, update.message.chat_id, stop_evt),
        daemon=True
    )
    thr.start()

    # get reply
    reply = chatbot_response(user_text)

    # stop typing indicator
    stop_evt.set()

    # send reply
    await update.message.reply_text(reply)

TOKEN = os.getenv("BOT_TOKEN", "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y")  # 🔑 safer with env var
PORT = int(os.environ.get("PORT", 10000))

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

application = Application.builder().token(TOKEN).build()

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to PharmaCare Bot!* \n\n"
    "I’m your friendly health & knowledge assistant. You can talk to me naturally —\n"
    "for example: *Tell me about malaria*, *Is it raining in Lagos?*, or *drug ibuprofen*.\n\n"
    "Below is a short guide so you know everything I can do and how to ask.\n\n"
    "— *Quick example commands*\n"
    "`/help` — show the full command list\n"
    "`wiki <topic>` — get a short summary from Wikipedia\n"
    "`drug <name>` — fetch drug info from medical APIs (OpenFDA / RxNav / DailyMed)\n"
    "`weather <city>` — get current weather for a city\n"
    "`time` — show current time (UTC/local)\n"
    "`news` — top headline\n"
    "`summarize <text>` — short summary (uses Hugging Face if configured)\n"
    "`expand <text>` — explain or expand text (uses Hugging Face if configured)\n"
    "`paraphrase <text>` — rephrase text (HF if available)\n"
    "`joke`, `cat fact`, `activity` — fun quick endpoints\n\n"
    "— *How I choose an API*\n"
    "For each question I try a single best API for that type of request (for speed and reliability).\n"
    "For example:\n"
    "• Weather → OpenWeather (if API key configured) → Open-Meteo → wttr.in\n"
    "• Drug info → OpenFDA → RxNav → DailyMed\n"
    "• Knowledge → Wikipedia → DuckDuckGo → Dictionary API\n\n"
    "— *Hugging Face (optional)*\n"
    "If the bot owner has set a Hugging Face API key, I will use it to:\n"
    "• understand complex English requests,\n"
    "• decide which tool to call when ambiguous,\n"
    "• rewrite long tool output into a natural human reply,\n"
    "• summarize/expand/paraphrase text.\n\n"
    "If HF is *not* configured, I still try the best web API and give helpful fallbacks.\n\n"
    "— *Privacy & Tokens*\n"
    "I never ask for tokens in chat. If an API key isn't working, the owner must set it in the server environment and restart the bot.\n\n"
    "If you want help or usage examples, type `/help`.\n",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ✅ uses HELP_TEXT from brain.py (always synced)
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")

# Chat handler using Brain
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    # show typing action before processing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # process with brain.py
    reply = chatbot_response(user_text)
    
    await update.message.reply_text(reply, parse_mode="Markdown")

# Handlers
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Run Telegram bot loop
loop = asyncio.new_event_loop()
def run_ptb():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    logger.info("✅ Telegram Bot Application started.")
    loop.run_forever()

Thread(target=run_ptb, daemon=True).start()

@app.route("/", methods=["GET"])
def home():
    return "🤖 Bot is alive!"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
    logger.info("✅ Update received and processed")
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)