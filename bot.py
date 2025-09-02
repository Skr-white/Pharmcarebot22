import os
import asyncio
import logging
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ======================
# CONFIG
# ======================
TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"
PORT = int(os.environ.get("PORT", 10000))  # Render expects 10000

# Flask app
app = Flask(__name__)

# Telegram app
application = Application.builder().token(TOKEN).build()

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

# ======================
# COMMAND HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("➡️ /start command triggered")
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("➡️ /help command triggered")
    await update.message.reply_text("You can use /start to begin or /help to see options.")

# Register handlers
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))

# ======================
# BACKGROUND EVENT LOOP
# ======================
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def start_bot():
    await application.initialize()
    await application.start()
    logger.info("✅ Telegram bot started")

loop.create_task(start_bot())

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET"])
def home():
    return "🤖 PharmaCare Bot is alive!", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        update = Update.de_json(request.get_json(force=True), application.bot)
        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        logger.info("✅ Update received and processed")
    except Exception as e:
        logger.error(f"❌ Error processing update: {e}")
    return "ok", 200

# Test route: send yourself a message
@app.route("/test", methods=["GET"])
def test():
    try:
        chat_id = 6224014992  # 👈 Your Telegram user ID
        asyncio.run_coroutine_threadsafe(
            application.bot.send_message(chat_id=chat_id, text="✅ Test message from Render!"),
            loop
        )
        return "Sent test message!", 200
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return "Test failed", 500

# ======================
# RUN FLASK
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)