import os
import asyncio
import logging
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from brain import chatbot_response, HELP_TEXT   # ✅ import HELP_TEXT from brain.py

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
        "👋 Hello! I am your PharmaCare Bot.\n"
        "Type naturally (e.g. *Tell me about malaria*) or use `/help` to see all commands.",
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