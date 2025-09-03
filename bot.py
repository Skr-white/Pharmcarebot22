‎import os
‎import asyncio
‎import logging
‎from threading import Thread
‎from flask import Flask, request
‎from telegram import Update
‎from telegram.ext import Application, CommandHandler, ContextTypes
‎
‎# ==============================
‎# Config
‎# ==============================
‎TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"
‎PORT = int(os.environ.get("PORT", 10000))
‎
‎# ==============================
‎# Logging
‎# ==============================
‎logging.basicConfig(level=logging.INFO)
‎logger = logging.getLogger("bot")
‎
‎# ==============================
‎# Flask App
‎# ==============================
‎app = Flask(__name__)
‎application = Application.builder().token(TOKEN).build()
‎
‎# ==============================
‎# Commands
‎# ==============================
‎async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
‎    logger.info(f"➡️ /start from {update.effective_user.id}")
‎    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")
‎
‎async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
‎    logger.info(f"➡️ /help from {update.effective_user.id}")
‎    await update.message.reply_text("You can use /start to begin or /help to see options.")
‎
‎application.add_handler(CommandHandler("start", start))
‎application.add_handler(CommandHandler("help", help_command))
‎
‎# ==============================
‎# Background Telegram loop
‎# ==============================
‎loop = asyncio.new_event_loop()
‎def run_ptb():
‎    asyncio.set_event_loop(loop)
‎    loop.run_until_complete(application.initialize())
‎    loop.run_until_complete(application.start())
‎    logger.info("✅ Telegram Bot Application started.")
‎    loop.run_forever()
‎
‎Thread(target=run_ptb, daemon=True).start()
‎
‎# ==============================
‎# Routes
‎# ==============================
‎@app.route("/", methods=["GET"])
‎def home():
‎    return "🤖 Bot is alive!"
‎
‎@app.route("/webhook", methods=["POST"])
‎def webhook():
‎    try:
‎        data = request.get_json(force=True)
‎        logger.info(f"📩 Incoming update: {data}")   # log full Telegram update
‎        update = Update.de_json(data, application.bot)
‎        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
‎        logger.info("✅ Update received and processed")
‎    except Exception as e:
‎        logger.error(f"❌ Error processing update: {e}")
‎    return "ok", 200
‎
‎# ==============================
‎# Run App
‎# ==============================
‎if __name__ == "__main__":
‎    app.run(host="0.0.0.0", port=PORT)
‎