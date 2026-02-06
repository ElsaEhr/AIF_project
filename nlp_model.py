# nlp_model.py
import torch
import torch.nn as nn

class TextClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes=11, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            300,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(300, num_classes)

    def forward(self, text, text_lengths):
        embedded = self.get_embeddings(text, text_lengths)
        return self.fc(embedded)

    def get_embeddings(self, text, text_lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]
