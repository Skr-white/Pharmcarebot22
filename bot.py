import os
import asyncio
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"
PORT = int(os.environ.get("PORT", 8080))

app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

# === Commands ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("➡️ /start command triggered")
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("➡️ /help command triggered")
    await update.message.reply_text("You can use /start to begin or /help to see options.")

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))

# === Run Telegram bot loop ===
loop = asyncio.new_event_loop()
def run_ptb():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    logger.info("✅ Telegram Bot Application started.")
    loop.run_forever()

Thread(target=run_ptb, daemon=True).start()

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return "🤖 Bot is alive!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        update = Update.de_json(request.get_json(force=True), application.bot)
        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        logger.info("✅ Update received and processed")
    except Exception as e:
        logger.error("❌ Error processing update: %s", str(e))
    return "ok", 200

# === Test endpoint ===
@app.route("/test", methods=["GET"])
def test():
    try:
        chat_id = 6220410492  # 👈 replace this with YOUR Telegram user ID
        asyncio.run_coroutine_threadsafe(
            application.bot.send_message(chat_id=chat_id, text="✅ Test message from Render!"),
            loop
        )
        return "Sent test message!", 200
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return "Test failed", 500

# === Run app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)