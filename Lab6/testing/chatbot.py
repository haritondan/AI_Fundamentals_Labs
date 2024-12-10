import string
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import torch
import pickle
from testing.model import Encoder, Decoder, Seq2Seq

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocabularies
with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)
    question_vocab = vocab_data["question_vocab"]
    answer_vocab = vocab_data["answer_vocab"]

# Load trained model and vocabularies
INPUT_DIM = len(question_vocab)
OUTPUT_DIM = len(answer_vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5

# Define the model
encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load model weights
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Define max lengths (Ensure these are consistent with your training)
max_question_len = 20  # Adjust based on your dataset
max_answer_len = 20    # Adjust based on your dataset

# Helper functions for text preprocessing
def tokenize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def preprocess_input(sentence, vocab, max_len):
    tokens = tokenize(sentence)
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    padded = indices + [0] * (max_len - len(indices))
    print(f"Original Sentence: {sentence}")
    print(f"Tokenized Sentence: {tokens}")
    print(f"Mapped Indices: {indices}")
    print(f"Padded Indices: {padded}")
    return torch.tensor([padded], dtype=torch.long, device=device)


def indices_to_sentence(indices, vocab):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return " ".join(reverse_vocab.get(idx, "<unk>") for idx in indices if idx not in {0, 1})

def beam_search_decoding(encoder, decoder, input_tensor, beam_width, max_len):
    """
    Perform beam search decoding.
    Args:
        encoder: Encoder model
        decoder: Decoder model
        input_tensor: Tensor of input indices
        beam_width: Number of beams to keep during decoding
        max_len: Maximum length of the output sequence
    Returns:
        Best sequence as a string (decoded)
    """
    hidden, cell = encoder(input_tensor)
    beams = [(0, [answer_vocab["<sos>"]], hidden, cell)]  # (score, sequence, hidden, cell)

    for _ in range(max_len):
        new_beams = []
        for score, seq, hidden, cell in beams:
            input_tensor = torch.tensor([seq[-1]], device=device)
            output, hidden, cell = decoder(input_tensor, hidden, cell)
            top_k = torch.topk(output.squeeze(), beam_width)

            for i in range(beam_width):
                token = top_k.indices[i].item()
                token_score = top_k.values[i].item()
                new_beams.append((score + token_score, seq + [token], hidden, cell))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # Stop if all beams generate <eos>
        if all(seq[-1] == answer_vocab["<eos>"] for _, seq, _, _ in beams):
            break

    # Return the sequence with the highest score
    best_sequence = beams[0][1]  # Top-scoring sequence
    return indices_to_sentence(best_sequence[1:-1], answer_vocab)  # Exclude <sos> and <eos>


# Generate model response
def generate_response(user_input):
    """
    Generate a response using beam search decoding.
    Args:
        user_input: String input from the user
    Returns:
        String response from the model
    """
    input_tensor = preprocess_input(user_input, question_vocab, max_question_len)
    response = beam_search_decoding(model.encoder, model.decoder, input_tensor, beam_width=3, max_len=max_answer_len)
    return response



# Telegram bot functions
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your chatbot. Ask me anything!")

async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    reply = generate_response(user_input)  # This remains synchronous
    await update.message.reply_text(reply)


def main():
    # Replace 'YOUR_TOKEN' with your actual bot token
    TOKEN = "7613500827:AAGqthzpgMk6opuaaR0Q538IzjOOid5TWOg"

    # Create the bot application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
