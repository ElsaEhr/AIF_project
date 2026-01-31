import argparse
import os
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

import torch.nn as nn
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassifier(nn.Module):

  def __init__(self, vocab_size, embedding_dim, num_classes=11):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
    self.lstm = nn.LSTM(embedding_dim,
                        300,
                        num_layers=2,
                        dropout=0.2,
                        batch_first=True)
    self.fc = nn.Linear(300, num_classes)

  def forward(self, text, text_lengths):
      embedded = self.get_embeddings(text, text_lengths)
      outputs=self.fc(embedded)
      return outputs


  def get_embeddings(self, text, text_lengths):
      embedded = self.embedding(text)
      #packed sequence
      packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
      _, (hidden, cell) = self.lstm(packed_embedded)
      return hidden[-1]



# Collate Function for DataLoader
def collate_batch(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    lengths = torch.tensor([len(x) for x in input_ids])

    input_ids = nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    return (
        input_ids.to(device),
        labels.to(device),
        lengths.to(device)
    )




from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=256):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.rename(columns={
            "movie_poster_path": "poster_path",
            "movie_plot": "plot",
            "movie_category": "label"})
        
        
        self.data['label'] = pd.Categorical(self.data['label']).codes
        self.tokenizer = tokenizer
        self.max_length = max_length



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            str(row["plot"]),
            truncation=True,
            padding=False,        # handled in collate_fn
            max_length=self.max_length,
            return_tensors=None
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'prediction genre from plot', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default = int(30), help='number of epochs')


    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr

    def train(model, dataloader, optimizer, criterion, epochs=5):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            t = tqdm(dataloader)
            for i, (text, labels, text_lengths) in enumerate(t):

                pred = model(text, text_lengths)
                loss = criterion(pred, labels)

                _, predicted = pred.max(1)
                running_corrects += predicted.eq(labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_description(f"epoch:{epoch} loss: {(running_loss / (i+1)):.4f} current accuracy:{round(running_corrects / total * 100, 2)}%")

    def test(model, dataloader):
        model.eval()
        test_corrects = 0
        total = 0
        with torch.no_grad():
            for text, labels, text_lengths in dataloader:
                pred = model(text, text_lengths).squeeze()
                _, predicted = pred.max(1)
                test_corrects += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return test_corrects / total




    # --- chemin vers dataset local ---
    dataset_path = './data/movie_plots.csv'  # chemin vers le dossier contenant les classes
    full_dataset = MovieDataset(dataset_path, tokenizer)



    # --- optionnel : split train/test ---
    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # --- dataloaders ---
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    net = TextClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=300,
        num_classes=10).to(device)
        # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{exp_name}')
        
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train(net, train_loader, optimizer, criterion, nb_epochs)
    test_acc = test(net, test_loader)

    print(f'test accuracy: {test_acc}')

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # sauvegarder les poids
    torch.save(net.state_dict(), './weights/text_classifier.pth')
    print("Poids sauvegardés !")