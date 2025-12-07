import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, num_classes=10, input_size=(3,128,128)):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # calcul automatique de la taille après conv+pool
        c, h, w = input_size
        h = (h - 4) // 2   # conv1 puis pool
        w = (w - 4) // 2
        h = (h - 4) // 2   # conv2 puis pool
        w = (w - 4) // 2
        self.flatten_size = 16 * h * w

        self.fc1 = nn.Linear(self.flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = torch.max_pool2d(x, (2,2))
        x = torch.flatten(x, 1) #or x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x


if __name__=='__main__':
    x = torch.rand(16,1,28,28)
    net = model()
    y = net(x)
    assert y.shape == (16,10)