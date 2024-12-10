import pandas as pd
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Tokenize function: splits a string into lowercase tokens
def tokenize(text):
    return text.lower().split()

# Build vocabulary from a dataset column
def build_vocab(texts, min_freq=1):
    token_counter = Counter()
    for text in texts:
        token_counter.update(tokenize(text))
    
    vocab = {word: idx + 4 for idx, (word, count) in enumerate(token_counter.items()) if count >= min_freq}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    vocab["<sos>"] = 2
    vocab["<eos>"] = 3
    return vocab


# Convert sentences to numerical sequences

def text_to_sequence(texts, vocab, add_sos_eos=False):
    sequences = []
    for text in texts:
        tokens = tokenize(text)
        sequence = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        if add_sos_eos:
            sequence = [vocab["<sos>"]] + sequence + [vocab["<eos>"]]
        sequences.append(sequence)
    return sequences


# Pad sequences to the same length
def pad_sequences(sequences, max_len):
    return np.array([seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences])

# Dataset class for PyTorch DataLoader
class QADataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.answers = torch.tensor(answers, dtype=torch.long)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# Decoder Network
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # Add sequence dimension
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.softmax(self.fc(outputs.squeeze(1)))
        return predictions, hidden, cell

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs

# Training function
def train(model, data_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for src, trg in data_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

# Validation function
def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # No teacher forcing
            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

# Load train and validation datasets
train_data = pd.read_csv("train.csv")  # Update the file path as needed
validation_data = pd.read_csv("validation.csv")  # Update the file path as needed

questions_train = train_data['question'].tolist()
answers_train = train_data['answer'].tolist()
questions_val = validation_data['question'].tolist()
answers_val = validation_data['answer'].tolist()


# Build vocabularies
question_vocab = build_vocab(questions_train + questions_val)
answer_vocab = build_vocab(answers_train + answers_val)

# Convert text to sequences
train_question_seqs = text_to_sequence(questions_train, question_vocab)
train_answer_seqs = text_to_sequence(answers_train, answer_vocab, add_sos_eos=True)
val_question_seqs = text_to_sequence(questions_val, question_vocab)
val_answer_seqs = text_to_sequence(answers_val, answer_vocab, add_sos_eos=True)

# Determine max sequence length for padding
max_question_len = max(len(seq) for seq in train_question_seqs + val_question_seqs)
max_answer_len = max(len(seq) for seq in train_answer_seqs + val_answer_seqs)

# Pad sequences
train_question_seqs = pad_sequences(train_question_seqs, max_question_len)
train_answer_seqs = pad_sequences(train_answer_seqs, max_answer_len)
val_question_seqs = pad_sequences(val_question_seqs, max_question_len)
val_answer_seqs = pad_sequences(val_answer_seqs, max_answer_len)



# Set parameters
INPUT_DIM = len(question_vocab)
OUTPUT_DIM = len(answer_vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
N_EPOCHS = 10
CLIP = 1
BATCH_SIZE = 32

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model setup
encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Load data
train_dataset = QADataset(train_question_seqs, train_answer_seqs)
val_dataset = QADataset(val_question_seqs, val_answer_seqs)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    val_loss = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch + 1}/{N_EPOCHS}")
    print(f"Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

torch.save(model.state_dict(), "best_model.pt")
import pickle

# Save vocabularies for reuse
with open("vocab.pkl", "wb") as f:
    pickle.dump({"question_vocab": question_vocab, "answer_vocab": answer_vocab}, f)
