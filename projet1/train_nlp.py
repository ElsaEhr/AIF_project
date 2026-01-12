import argparse
import os
from statistics import mean

import torch

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from model import poster_classifier
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TextClassifier(nn.Module):

  def __init__(self, vocab_size, embedding_dim, output_dim=4):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.lstm = nn.LSTM(embedding_dim,
                        100,
                        num_layers=2,
                        dropout=0.2,
                        batch_first=True)
    self.fc = nn.Linear(100, output_dim)

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
    label_list, text_list, lengths = [], [], []
    for example in batch:
        label = example["label"]
        tokens = example["tokens"]
        tensor = text_to_tensor(tokens)
        label_list.append(label)
        text_list.append(tensor)
        lengths.append(len(tensor))
    text_tensor = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    return text_tensor.to(device), label_tensor.to(device), lengths_tensor.to(device)


from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, names=["poster_path", "plot", "label"])
        # Ensure labels are integers
        self.data['label'] = pd.Categorical(self.data['label']).codes
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Using .iloc ensures we get the N-th row regardless of the index label
        row = self.data.iloc[idx]
        tokens = str(row['plot']).lower().split() # Simple whitespace tokenization
        return {
            "tokens": tokens,
            "label": row['label']
        }



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'prediction genre', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default = int(10), help='number of epochs')


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

                pred = model(text, text_lengths).squeeze() #convert to 1D tensor
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

    # --- transformations ---
    transform = transforms.Compose([
        transforms.Resize((128,128)),  # adapte selon ton besoin
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # --- chemin vers dataset local ---
    dataset_path = './data/movie_plots.csv'  # chemin vers le dossier contenant les classes
    full_dataset = MovieDataset(dataset_path)



    # --- optionnel : split train/test ---
    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # --- dataloaders ---
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    net = poster_classifier().to(device)
        # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{exp_name}')
        
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train(net, train_loader, optimizer, nn.CrossEntropyLoss(), nb_epochs)
    test_acc = test(net, test_loader)

    print(f'test accuracy: {test_acc}')

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # sauvegarder les poids
    #torch.save(net.state_dict(), './weights/poster_classifier.pth')
    #print("Poids sauvegardés !")