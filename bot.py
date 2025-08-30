import os
import asyncio
from threading import Thread
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"
PORT = int(os.environ.get("PORT", 8080))

app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can use /start to begin or /help to see options.")

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))

# Run Telegram in background loop
loop = asyncio.new_event_loop()
def run_ptb():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.start())
    print("✅ Telegram Bot Application started.")
    loop.run_forever()

Thread(target=run_ptb, daemon=True).start()

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
    @app.route("/", methods=["GET"])
def home():
    return "🤖 Bot is alive!"
