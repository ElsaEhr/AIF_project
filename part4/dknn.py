import argparse
import json
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
from torchvision.datasets import ImageFolder

from test_functions import DKNN, compute_features

from model_resNet import poster_classifier

import pickle

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#chargement du modèle
net = poster_classifier().to(device)
state_dict = torch.load('../weights_resNet/poster_classifier.pth',weights_only=True)
net.load_state_dict(state_dict)



#transformations
transform = transforms.Compose([
            transforms.Resize((224,224)),  # adapte selon ton besoin
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])




# chargement des données
dataset_path = '../content/' 
full_dataset = ImageFolder(root=dataset_path, transform=transform)

dataset_full = ImageFolder(root=dataset_path, transform=transform)

total_size = len(dataset_full)
indices = list(range(total_size))
split = int(0.8 * total_size)
np.random.shuffle(indices)

train_idx, test_idx = indices[:split], indices[split:]

total_size = len(train_idx)
split = int(0.8 * total_size)
np.random.shuffle(train_idx)

train_idx, val_idx = train_idx[:split], train_idx[split:]
trainset = torch.utils.data.Subset(dataset_full, train_idx)
testset = torch.utils.data.Subset(dataset_full, test_idx)

if __name__ == "__main__":

   


    #intialisation DKNN
    dknn = DKNN(k=20)
    fit_features = compute_features(trainset, net, device)
    dknn.fit(fit_features)
    threshold=0.7579877 #seuil calculé dans ood_train_and_test.py

    #sauvegarde
    model_data = {
        "dknn": dknn,
        "threshold": threshold,
        "testset" : testset
    }
    with open("dknn.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Modèle sauvegardé dans dknn_model.pkl")




