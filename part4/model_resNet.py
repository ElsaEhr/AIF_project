import torch
import torch.nn as nn
from torchvision import models

class poster_classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(poster_classifier, self).__init__()
        
        # Chargement ResNet18 pré-entraîné
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        #gel des couches de resNet
        for param in self.resnet.parameters():
            param.requires_grad = False

        #Dégel des couches 3 et 4 pour les adapter à notre étude
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True


        #défition de notre classifieur
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )



    def forward(self, x, return_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            return features

        return self.resnet.fc(features)