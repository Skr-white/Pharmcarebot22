import os
import asyncio
import logging
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"
PORT = int(os.environ.get("PORT", 8080))

app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

# Commands
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"➡️ /start command from {update.effective_user.id}")
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"➡️ /help command from {update.effective_user.id}")
    await update.message.reply_text("You can use /start to begin or /help to see options.")

# Run Telegram bot loop
loop = asyncio.new_event_loop()
def run_ptb():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    logger.info("✅ Telegram Bot Application started.")
    loop.run_forever()

Thread(target=run_ptb, daemon=True).start()

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        update = Update.de_json(data, application.bot)
        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        logger.info(f"✅ Update received and processed: {data}")
    except Exception as e:
        logger.error(f"❌ Error processing update: {e}")
    return "ok", 200