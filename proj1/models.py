"""
Defines the model classes using PyTorch
"""
import torch
from torch import nn as nn
from torch.nn import functional as F


class CNN_1(nn.Module):
    def __init__(self, nb_channels_1=32, kernel_1=3, nb_channels_2=64, kernel_2=3, nb_hidden_1 = 128, nb_hidden_2 = 10, dropout_p=0.0, padding=True, double_conv=False, batchNorm=False):
        super().__init__()
        
        padding_1=(kernel_1-1)//2 if padding else 0
        padding_2=(kernel_2-1)//2 if padding else 0
        pool = 2
        d1 = (14-kernel_1+1 +2*padding_1)//pool
        d2 = (d1-kernel_2+1 +2*padding_2)//pool
        self.nb_features = nb_channels_2*d2*d2
#         print('Number of features = ', self.nb_features)

        feaure_layers = []
        feaure_layers.append(nn.Conv2d(2, nb_channels_1, kernel_size=kernel_1, padding=padding_1))
        feaure_layers.append(nn.ReLU())
        if batchNorm:
            feaure_layers.append(nn.BatchNorm2d(nb_channels_1))
        if double_conv:
            feaure_layers.append(nn.Conv2d(nb_channels_1, nb_channels_1, kernel_size=kernel_1, padding=padding_1))
            feaure_layers.append(nn.ReLU())
            if batchNorm:
                feaure_layers.append(nn.BatchNorm2d(nb_channels_1))
        feaure_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        feaure_layers.append(nn.Dropout2d(p=dropout_p))

        feaure_layers.append(nn.Conv2d(nb_channels_1, nb_channels_2, kernel_size=kernel_2, padding=padding_2))
        feaure_layers.append(nn.ReLU())
        if batchNorm:
            feaure_layers.append(nn.BatchNorm2d(nb_channels_2))
        if double_conv:
            feaure_layers.append(nn.Conv2d(nb_channels_2, nb_channels_2, kernel_size=kernel_2, padding=padding_2))
            feaure_layers.append(nn.ReLU())
            if batchNorm:
                feaure_layers.append(nn.BatchNorm2d(nb_channels_2))
        feaure_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        feaure_layers.append(nn.Dropout2d(p=dropout_p))

        feaure_layers.append(nn.Flatten())
        self.features = nn.Sequential(*feaure_layers)

        classifier_layers = []
        classifier_layers.append(nn.Linear(in_features=self.nb_features, out_features=nb_hidden_1))
        classifier_layers.append(nn.ReLU())
        if batchNorm:
            classifier_layers.append(nn.BatchNorm1d(nb_hidden_1))
        classifier_layers.append(nn.Dropout2d(p=dropout_p))

        classifier_layers.append(nn.Linear(in_features=nb_hidden_1, out_features=nb_hidden_2))
        classifier_layers.append(nn.ReLU())
        if batchNorm:
            classifier_layers.append(nn.BatchNorm1d(nb_hidden_2))
        classifier_layers.append(nn.Dropout2d(p=dropout_p))

        classifier_layers.append(nn.Linear(in_features=nb_hidden_2, out_features=2))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return  x


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

