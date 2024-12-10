from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackContext

from utils import generate_answer

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your chatbot. Ask me anything!")

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_message = update.message.text
        response = generate_answer(user_message)

        print(f"{update.message.from_user.username}: {user_message}")
        print(f"Response: {response}")

        await update.message.reply_text(response)
    except Exception as e:
        print(e)
        await update.message.reply_text("Sorry, I don't know.")


app = ApplicationBuilder().token("your-token").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(callback=respond, filters=None))

app.run_polling()
