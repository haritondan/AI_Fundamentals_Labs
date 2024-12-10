from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext
import torch
import spacy

# Load your trained model (update with your actual model loading code)
def load_model():
    # Assuming your model is a PyTorch model, load it here.
    model_path = "seq2seq.pth"  # Update this
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Preprocessing function (adjust based on your model)
def preprocess_input(user_input):
    # Example preprocessing
    nlp = spacy.load("en_core_web_sm")
    processed = [token.text for token in nlp(user_input)]
    return processed

# Generate model response
def generate_response(user_input):
    preprocessed_input = preprocess_input(user_input)
    # Convert preprocessed input to model-compatible format
    # Example placeholder: Adjust based on your model's input format
    input_tensor = torch.tensor([preprocessed_input])
    
    with torch.no_grad():
        output = model(input_tensor)
    return output

# Telegram bot handlers
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Hi! Send me a message and I'll process it using the model.")

def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    try:
        response = generate_response(user_message)
        update.message.reply_text(f"Model response: {response}")
    except Exception as e:
        update.message.reply_text(f"Error processing message: {e}")

def main():
    # Replace 'YOUR_BOT_TOKEN' with your bot's token from BotFather
    updater = Updater("7613500827:AAGqthzpgMk6opuaaR0Q538IzjOOid5TWOg")
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(filters.text & ~filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
