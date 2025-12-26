import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from sklearn.metrics import roc_auc_score, precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from model_resNet import poster_classifier

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from test_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

def test_model(model, dataloader,device,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy =  correct / total
    avg_test_loss = test_loss / len(dataloader)

    
    
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {100*accuracy:.2f}%")
    return accuracy

def train_model(model,epochs,train_loader,val_loader,device,criterion,optimizer,writer):#,scheduler):
    
    best_acc=0
    for ep in range (epochs):
        model.train()
        total_loss=0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        #scheduler.step()
        print(f"Epoch [{ep + 1}/{epochs}], Loss: {avg_loss:.4f}")
        for i, group in enumerate(optimizer.param_groups):
            print(f"LR group {i}: {group['lr']}")
        accuracy=test_model(model, val_loader,device,criterion)
        writer.add_scalar('Loss/train', avg_loss, ep)
        if (accuracy>best_acc):
            # sauvegarder les poids
            best_acc=accuracy
            torch.save(model.state_dict(), './weights_resNet/poster_classifier.pth')
            print("Poids sauvegardés")
            # for experiment management









if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'prediction genre', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(2*1e-4), help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default = int(5),help='number of epochs')


    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr

    #writer pour le tensor board
    writer = SummaryWriter(f'runs/{args.exp_name}')

    #transformations
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # adapte selon ton besoin
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #data augmentation pour le train
    transform_data_augmentation = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



    dataset_path = './content/' 
    anomalies_path = './flickr8k/'
    full_dataset = ImageFolder(root=dataset_path, transform=transform)
    anomalies_dataset = ImageFolder(root=anomalies_path, transform=transform)


    #print(full_dataset.classes)
    #print(full_dataset.class_to_idx)



    
    """
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])
    trainset, valset = random_split(trainset, [0.9, 0.1])
    
    """

    augmented_dataset_full = ImageFolder(root=dataset_path, transform=transform_data_augmentation)
    dataset_full = ImageFolder(root=dataset_path, transform=transform)

    #division en train, validation et test, sans data augmentation sur  validation et test
    total_size = len(augmented_dataset_full)
    indices = list(range(total_size))
    split = int(0.8 * total_size)
    np.random.shuffle(indices)

    train_idx, test_idx = indices[:split], indices[split:]

    total_size = len(train_idx)
    split = int(0.8 * total_size)
    np.random.shuffle(train_idx)

    train_idx, val_idx = train_idx[:split], train_idx[split:]

    """
    Vérification que les ensembles sont bien disjoints : ok
    print(set(train_idx) & set(test_idx))
    print(set(train_idx) & set(val_idx))
    print(set(val_idx) & set(test_idx))
    """
    trainset = torch.utils.data.Subset(augmented_dataset_full, train_idx)
    valset = torch.utils.data.Subset(dataset_full, val_idx)
    testset = torch.utils.data.Subset(dataset_full, test_idx)

    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,  num_workers=2)
    valoader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    anomalies_test,_ = torch.utils.data.random_split(anomalies_dataset, [10_000, len(anomalies_dataset) - 10_000])
    anomalies_testloader = DataLoader(anomalies_test, batch_size=batch_size, shuffle=False)
    
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of test samples: {len(testset)}")
    print(f"Number of anomalies test samples: {len(anomalies_test)}")


    net = poster_classifier().to(device)


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    

    if not os.path.exists('./weights_resNet'): #entrainement du modèle si les poids ne sont pas enregistrés

        os.makedirs('./weights_resNet')

        #pour resNet, lr réglé en fonction de la couche
        optimizer = optim.Adam([
            {'params': net.resnet.layer3.parameters(), 'lr': 1e-6,'weight_decay':1e-4},
            {'params': net.resnet.layer4.parameters(), 'lr': 1e-5,'weight_decay':1e-4}, 
            {'params': net.resnet.fc.parameters(), 'lr': lr,'weight_decay':1e-4} ])     
        
        #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)

        train_model(net, nb_epochs,trainloader,valoader,device,criterion,optimizer,writer)#,scheduler)
        
        class_names = full_dataset.classes

        cm,test_acc = plot_confusion_matrix(
                model=net,
                dataloader=testloader,
                criterion=criterion,
                device=device,
                class_names=class_names
        )
        

        writer.add_hparams({'lr': lr, 'bsize': batch_size}, {'hparam/accuracy': test_acc}, run_name='posters_classif')


        #add embeddings to tensorboard
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)

        # On utilise le modèle pour extraire les caractéristiques (features)
        with torch.no_grad():
            embeddings = net(images, return_features=True)

        # Ajout à TensorBoard
        writer.add_embedding(embeddings,
                            metadata=[full_dataset.classes[l] for l in labels],
                            label_img=images, 
                            global_step=1)

        # --- SAUVEGARDE DU GRAPHE ET D'UNE GRILLE D'IMAGES ---
        writer.add_graph(net, images)
        img_grid = make_grid(images)
        writer.add_image('poster_samples', img_grid)


    else : #sinon on charge les poids enregistrés
        print("Chargement des poids existants...")
        state_dict = torch.load('./weights_resNet/poster_classifier.pth',weights_only=True)
        
        net.load_state_dict(state_dict)

        class_names = full_dataset.classes

        cm,test_acc = plot_confusion_matrix(
                model=net,
                dataloader=testloader,
                criterion=criterion,
                device=device,
                class_names=class_names
        )
        
        


    net.eval()

    #SPOILER : MEILLEURES SCORING FUNCTIONS : 1/DKNN 2/MAHALANOBIS 3/ENERGY

    #LOGIT BASED SCORES---------------------------------------------------------------------------------------------

    test_logits_negatives = compute_logits(testloader, net, device)
    test_logits_positives = compute_logits(anomalies_testloader, net, device)

    scoring_functions = {
    'MLS': mls,
    'MSP': msp,
    'Energy': energy,
    'Entropy': entropy
    }
    metrics_dict = {}

    for method, scoring_function in scoring_functions.items():
        
        # Compute scores
        scores_negatives = scoring_function(test_logits_negatives)
        scores_positives = scoring_function(test_logits_positives)

        # Plot histogram of scores
        plot_scores(scores_positives,scores_negatives,method,writer)

        metrics_dict[method] = {}

        # Plot ROC curve and compute AUROC
        auroc = roc_auc(scores_negatives, scores_positives)
        metrics_dict[method]['auroc'] = auroc

        # Compute threshold for the given target_tpr
        threshold = compute_threshold(scores_positives, target_tpr=0.95)

        # Compute and store remaining metrics
        metrics_dict[method]['accuray'] = accuracy(scores_negatives, scores_positives, threshold)
        metrics_dict[method]['tpr'], metrics_dict[method]['fpr'] = tpr_fpr(scores_negatives, scores_positives, threshold)
        metrics_dict[method]['precision'], metrics_dict[method]['recall'] = precision_recall(scores_negatives, scores_positives, threshold)
        metrics_dict[method]['f1'] = f_beta(scores_negatives, scores_positives, threshold, beta=1)

        writer.add_scalar(f'Metrics/{method}/AUROC', auroc, 0)
        writer.add_scalar(f'Metrics/{method}/F1_Score', metrics_dict[method]['f1'], 0)
 


    #FEATURE BASED SCORES----------------------------------------------------------------------------------
    target_tpr = 0.9
    metrics_dict = {}


    fit_features = compute_features(trainset, net, device)
    test_features_negatives = compute_features(testset, net, device)
    test_features_positives = compute_features(anomalies_test, net, device)
    fit_features = fit_features / fit_features.norm(dim=1, keepdim=True)
    test_features_negatives = test_features_negatives / test_features_negatives.norm(dim=1, keepdim=True)
    test_features_positives = test_features_positives / test_features_positives.norm(dim=1, keepdim=True)

    #DKNN------------------------------------------

    metrics_dict['DKNN'] = {}

    dknn = DKNN(k=20)
    dknn.fit(fit_features)

    scores_negatives = dknn.compute_scores(test_features_negatives)
    scores_positives = dknn.compute_scores(test_features_positives)

    # Plot the histogram of the scores
    plot_scores(scores_positives,scores_negatives,"DKNN",writer)

    auroc = roc_auc(scores_negatives, scores_positives)
    metrics_dict['DKNN']['auroc'] = auroc

    threshold = compute_threshold(scores_positives, target_tpr)

    metrics_dict['DKNN']['accuray'] = accuracy(scores_negatives, scores_positives, threshold)
    metrics_dict['DKNN']['tpr'], metrics_dict['DKNN']['fpr'] = tpr_fpr(scores_negatives, scores_positives, threshold)
    metrics_dict['DKNN']['precision'], metrics_dict['DKNN']['recall'] = precision_recall(scores_negatives, scores_positives, threshold)
    metrics_dict['DKNN']['f1'] = f_beta(scores_negatives, scores_positives, threshold, beta=1)

    #MAHALANOBIS--------------------------------------

    metrics_dict['Mahalanobis'] = {}

    train_targets = np.array(trainset.dataset.targets)[trainset.indices]

    maha = Mahalanobis()
    maha.fit(fit_features, train_targets)

    scores_negatives = maha.compute_scores(test_features_negatives)
    scores_positives = maha.compute_scores(test_features_positives)

    # Plot the histogram of the scores
    plot_scores(scores_positives,scores_negatives,"mahalanobis",writer)

    auroc = roc_auc(scores_negatives, scores_positives)
    metrics_dict['Mahalanobis']['auroc'] = auroc

    threshold = compute_threshold(scores_positives, target_tpr)

    metrics_dict['Mahalanobis']['accuray'] = accuracy(scores_negatives, scores_positives, threshold)
    metrics_dict['Mahalanobis']['tpr'], metrics_dict['Mahalanobis']['fpr'] = tpr_fpr(scores_negatives, scores_positives, threshold)
    metrics_dict['Mahalanobis']['precision'], metrics_dict['Mahalanobis']['recall'] = precision_recall(scores_negatives, scores_positives, threshold)
    metrics_dict['Mahalanobis']['f1'] = f_beta(scores_negatives, scores_positives, threshold, beta=1)


#to look at the tensorboard : tensorboard --logdir runs