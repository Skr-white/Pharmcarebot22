from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = "8282174001:AAF1ef9UK0NUdUa3fJTpmU0Q1drPp0IIS0Y"  # your bot token

# Flask app
app = Flask(__name__)

# Create Telegram Application once (global)
application = Application.builder().token(TOKEN).build()

# Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! I am your PharmaCare Bot. How can I help you today?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can use /start to begin or /help to see options.")

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))

# Webhook endpoint
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    application.update_queue.put(update)
    return "ok"

# Flask main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    
