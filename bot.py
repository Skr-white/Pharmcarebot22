import os
import asyncio
import logging
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from brain import chatbot_response  # <-- our smart brain

TOKEN= "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"  # replace with your bot token
PORT = int(os.environ.get("PORT", 10000))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. Type 'wiki <topic>', 'drug <name>', or just chat with me!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can try:\n- `wiki diabetes`\n- `drug ibuprofen`\n- `search healthy diet`\nOr just say hi!")

# Chat handler using Brain
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    reply = chatbot_response(user_text)
    await update.message.reply_text(reply)

# Add handlers
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