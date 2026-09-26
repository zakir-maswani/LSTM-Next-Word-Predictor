"""
LSTM Model
----------
Simple LSTM based Next Word Prediction model.

How it works (in plain words):
1. Embedding layer turns each word index into a vector of numbers.
2. LSTM layer reads the sequence of vectors and remembers context.
3. Linear (fully connected) layer converts the LSTM's memory into
   scores for every word in our vocabulary.
4. The word with the highest score is our predicted "next word".
"""

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        # we only need the final hidden state to predict the next word
        _, (final_hidden_state, _) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output
