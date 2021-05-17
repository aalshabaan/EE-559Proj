"""
Defines the model classes using PyTorch
"""
import torch
from torch import nn as nn
from torch.nn import functional as F


class Net1(nn.Module):
    def __init__(self, nb_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x


class Net2(nn.Module):
    def __init__(self, nb_hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(1024, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        #         x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        #         x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2, padding=1)
        x = F.relu(self.fc1(x.view(-1, 1024)))
        x = self.fc2(x)
        return x


class Net3(nn.Module):
    def __init__(self, nb_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size,
                 skip_connections=True, batch_normalization=True):
        super().__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization

    def forward(self, x):
        y = F.dropout2d(x, 0.1)
        y = self.conv1(y)
        if self.batch_normalization: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = F.dropout2d(y, 0.1)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.relu(y)

        return y


class ResNet(nn.Module):

    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size, nb_classes=10, nb_pred=2,
                 skip_connections=True, batch_normalization=True,
                 auxiliary_loss=False, auxiliary_weight=0.4):
        super().__init__()

        self.conv = nn.Conv2d(2, nb_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size, skip_connections, batch_normalization)
              for _ in range(nb_residual_blocks))
        )

        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight
        if auxiliary_loss:
            self.fc1 = nn.Linear(nb_channels, nb_classes * nb_pred)
            self.fc2 = nn.Linear(nb_classes * nb_pred, nb_pred)
        else:
            self.fc = nn.Linear(nb_channels, nb_pred)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.resnet_blocks(x)
        x = F.avg_pool2d(x, 14).view(x.size(0), -1)
        if self.auxiliary_loss:
            x = self.fc1(x)
            classes = x.view(-1, 10, 2)
            x = F.relu(x)
            x = self.fc2(x)
            return x, classes
        else:
            x = self.fc(x)
            return x

class SiameseNet1(nn.Module):
    """
    Siamese model with possible auxiliary losses that are used to train both the convolutional layers and the hidden fully connected layers
    """
    def __init__(self, auxiliary_loss=False, auxiliary_weight=0.4):
        super().__init__()
        self.shared_model = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), padding=(2,2)),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=10),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=10)
        )
        self.comparator = nn.Sequential(
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2)
        )
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight


    def forward(self, x):
        num_1 = x[:,0,:,:].unsqueeze(1)
        num_2 = x[:,1,:,:].unsqueeze(1)

        pred_1 = self.shared_model(num_1)
        pred_2 = self.shared_model(num_2)
        if self.auxiliary_loss:
            classes = torch.cat([pred_1.view(-1,10,1), pred_2.view(-1,10,1)], dim=2)
            return self.comparator(torch.cat((pred_1, pred_2), 1)),  classes
        else:
            return self.comparator(torch.cat((pred_1, pred_2), 1))



class SiameseNet2(nn.Module):
    def __init__(self, auxiliary_loss=False, auxiliary_weight=0.4):
        super().__init__()
        self.shared_model = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), padding=(2,2)),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.ReLU(),

        )


        self.comparator = nn.Sequential(
            nn.Linear(in_features=3*3*2*64, out_features=128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=2),
            nn.ReLU()
        )
        if auxiliary_loss:
            self.class_predictor = nn.Sequential(
                nn.Linear(3*3*64, 128),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128,10)
            )

        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight

    def forward(self, x):
        num_1 = x[:,0,:,:].unsqueeze(1)
        num_2 = x[:,1,:,:].unsqueeze(1)

        pred_1 = self.shared_model(num_1)
        pred_2 = self.shared_model(num_2)

        out = self.comparator(torch.cat((pred_1, pred_2), 1))
        if self.auxiliary_loss:
            class_1 = self.class_predictor(pred_1).unsqueeze(2)
            class_2 = self.class_predictor(pred_2).unsqueeze(2)
            classes = torch.cat([class_1, class_2],2)

            return out, classes
        else:
            return out

