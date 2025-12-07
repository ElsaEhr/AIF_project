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

from model import model
#use it if you have module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem' error
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

 # setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        if writer is not None:
            writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default = int(10), help='number of epochs')


    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    
    
    
        # --- transformations ---
    transform = transforms.Compose(
        [transforms.Resize((128,128)),  # adapte selon ton besoin
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    # --- chemin vers ton dataset local ---
    dataset_path = './content'  # chemin vers le dossier contenant les classes
    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    # --- optionnel : split train/test ---
    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # --- dataloaders ---
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = model().to(device)
        # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{exp_name}')
        
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train(net, optimizer, trainloader, nb_epochs, writer)
    test_acc = test(net, testloader)

    print(f'test accuracy: {test_acc}')


    # --- transformations ---
    transform = transforms.Compose([
        transforms.Resize((128,128)),  # adapte selon ton besoin
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # --- chemin vers ton dataset local ---
    dataset_path = './content'  # chemin vers le dossier contenant les classes
    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    # --- optionnel : split train/test ---
    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # --- dataloaders ---
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = model().to(device)
        # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{exp_name}')
        
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train(net, optimizer, trainloader, nb_epochs, writer)
    test_acc = test(net, testloader)

    print(f'test accuracy: {test_acc}')

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # sauvegarder les poids
    torch.save(net.state_dict(), './weights/mnist_net.pth')
    print("Poids sauvegardés !")