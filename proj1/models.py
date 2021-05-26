"""
Defines the model classes using PyTorch
"""
import torch
from torch import nn as nn
from torch.nn import functional as F



class CNN_1(nn.Module):
    """
    This class is used to construct both the baseline and the optimized versions of convolutional neural networks descripted in the report.
    """
    def __init__(self, nb_channels_1=32, kernel_1=3, nb_channels_2=64, kernel_2=3, nb_hidden_1 = 128, nb_hidden_2 = 10, dropout_p=0.0, padding=True, double_conv=False, batchNorm=False):
        """
        Constructs a convolutional neural network with two convolutional layers and three hidden fully-connected layers which result to an output of two units. This model has the functionality to determine automatically the number of input features for the fully connected layers according to the choice of parameters, such as the size of kernel and channels. Additionally, the model includes dropout layers after each major network unit and batch normalization after ReLU activations. Finally, it provides an option to replace each convolution layer with two convolutions where the second one preserves the number of channels and uses the same size of kernel as the first one.
        
        The input layer takes tensors of size (batch_size, 2, 14, 14) and returns a tensor of size (batch_size, 2)
        :param nb_channels_1: Int: Number of output channels for the first convolutional layer.
        :param kernel_1: Int: Size of kernel for the first convolutional layer.
        :param nb_channels_2: Int: Number of output channels for the second convolutional layer.
        :param kernel_2: Int: Size of kernel for the second convolutional layer.
        :param nb_hidden_1: Int: Number of output units for the first hidden layer.
        :param nb_hidden_2: Int: Number of output units for the second hidden layer.
        :param dropout_p: float: Probability of an element to be zeroed in the dropout layer.
        :param padding: bool: If true preserves the dimensionality of the input signal through the convolution layers.
        :param double_conv: bool: If true replaces each convolution layer with two convolutions where the second one preserves the number of channels and uses the same size of kernel as the first one.
        :param batchNorm: bool: If true adds batch normalization after ReLU activations.
        :return: Tensor, with the comparison prediction classes of size (batch_size, 2).
        """
        super().__init__()
        
        #compute padding for each convolution to preserve the dimensionality of input signal
        padding_1=(kernel_1-1)//2 if padding else 0
        padding_2=(kernel_2-1)//2 if padding else 0
        
        #downsampling size of max pooling layers
        pool = 2
        
        #compute the number of features for the input of fully connected layers 
        d1 = (14-kernel_1+1 +2*padding_1)//pool
        d2 = (d1-kernel_2+1 +2*padding_2)//pool
        self.nb_features = nb_channels_2*d2*d2

        #1) Feature Extraction Part
        #contains the layers for the feature extraction part of the classifier 
        feaure_layers = []
        
        #first convolutional layer
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

        #second convolutional layer
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

        #feature extractor output
        feaure_layers.append(nn.Flatten())
        self.features = nn.Sequential(*feaure_layers)

        #2) Classifier Part
        #contains the layers for the classification part of the classifier
        classifier_layers = []
        
        #fisrt hidden layer
        classifier_layers.append(nn.Linear(in_features=self.nb_features, out_features=nb_hidden_1))
        classifier_layers.append(nn.ReLU())
        if batchNorm:
            classifier_layers.append(nn.BatchNorm1d(nb_hidden_1))
        classifier_layers.append(nn.Dropout2d(p=dropout_p))
        
        #second hidden layer
        classifier_layers.append(nn.Linear(in_features=nb_hidden_1, out_features=nb_hidden_2))
        classifier_layers.append(nn.ReLU())
        if batchNorm:
            classifier_layers.append(nn.BatchNorm1d(nb_hidden_2))
        classifier_layers.append(nn.Dropout2d(p=dropout_p))
        
        #third hidden layer
        classifier_layers.append(nn.Linear(in_features=nb_hidden_2, out_features=2))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: Tensor: input tensor
        :return: Tensor, with the comparison prediction classes
        """
        x = self.features(x)
        x = self.classifier(x)
        return  x


    
class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size,
                 skip_connections=True, batch_normalization=True):
        """
        The input layer takes tensors of size (batch_size, nb_channels, 14-kernel_size+1, 14-kernel_size+1), output is a tensor of size (batch_size, nb_channels, 14-kernel_size+1, 14-kernel_size+1)
        :param nb_channels: Int: Number of input and output channels for convolutional layers.
        :param kernel_size: Int: Size of kernel for convolutional layers.
        :param skip_connections: bool: if true skips connection.
        :param batch_normalization: bool: if true adds batch normalization.
        :return: a tensor of size (batch_size, nb_channels, 14-kernel_size+1, 14-kernel_size+1)
        """
        super().__init__()
        
        #first convolutional layer
        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)
        
        #batch normalization
        self.bn1 = nn.BatchNorm2d(nb_channels)
           
        #second convolutional layer
        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)
        
        #batch normalization
        self.bn2 = nn.BatchNorm2d(nb_channels)

        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization

    def forward(self, x):
        """
        Forward pass of the ResNetBlock
        :param x: Tensor: input tensor
        :return: Tensor: output tensor
        """
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
    """
    implementation of residual neural network.
    """

    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size, nb_classes=10, nb_pred=2,
                 skip_connections=True, batch_normalization=True,
                 auxiliary_loss=False, auxiliary_weight=0.4):
        """
        The input layer takes tensors of size (batch_size, 2, 14, 14), main output is a tensor of size (batch_size, 2)
        :param nb_residual_blocks: Int: Number of residual blocks.
        :param nb_channels: Int: Number of input(except the first one) and output channels for convolutional layers.
        :param kernel_size: Int: Size of kernel for convolutional layers.
        :param nb_classes: Int: Number of classes that the digits belong to. Used if auxiliary_loss equal true.
        :param nb_pred: Int: Number of outputs to predict.
        :param skip_connections: bool: if true skip connection.
        :param batch_normalization: bool: if true adds batch normalization in residual blocks.
        :param auxiliary_loss: bool: Whether to use auxiliary losses, if true the model returns 2 predictions, output and class.
        :param auxiliary_weight: float: The weight of the auxiliary loss compared to the main loss which has a weight of 1.
        :return: if `auxiliary_loss`: returns a tuple of tensors which are (batch_size, 2), (batch_size, 10, 2).
        The first has predictions of whether the first digit is smaller than the latter, the second has the digit class prediction for each input channel.
        otherwise returns only the first tensor of size (batch_size,2)
        """
        super().__init__()

        self.conv = nn.Conv2d(2, nb_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)
        
        #sequential of ResNetBlocks
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
        """
        Forward pass of the model
        :param x: Tensor: input tensor
        :return: *Tensor, comparison predictions and, if self.auxiliary_loss, digit predictions
        """
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
    Siamese model with possible auxiliary losses that are used to train both the convolutional layers and the hidden
    fully connected layers doing the digit classification.
    """
    def __init__(self, auxiliary_loss=False, auxiliary_weight=0.4):
        """
        Construct a siamese CNN with two convolutional layers, two hidden fully-connected layers, and an output layer.
        This model shares weights across both input channels, and uses an optional auxiliary loss to train everything before the ouput layer
        The input layer takes tensors of size (batch_size, 2, 14, 14), the main output is of size (batch_size, 2)
        :param auxiliary_loss: bool: Whether to use auxiliary losses, if true the model returns 2 predictions, output and class
        :param auxiliary_weight: float: The weight of the auxiliary loss compared to the main loss which has a weight of 1
        :return: if `auxiliary_loss`: returns a tuple of tensors which are (batch_size, 2), (batch_size, 10, 2).
        The first has predictions of whether the first digit is smaller than the latter, the second has the digit class prediction for each input channel.
        otherwise returns only the first tensor of size (batch_size,2)
        """
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
        """
        Forward pass of the model
        :param x: Tensor: input tensor
        :return: *Tensor, comparison predictions and, if self.auxiliary_loss, digit predictions
        """
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
    """
        Siamese model with possible auxiliary losses that are used to train only the convolutional layers.
        """
    def __init__(self, auxiliary_loss=False, auxiliary_weight=0.4):
        """Construct a siamese CNN with two convolutional layers, two hidden fully-connected layers, and an output layer.
        This model shares weights across both input channels, and uses an optional auxiliary loss to train the convolutional layers only
        The input layer takes tensors of size (batch_size, 2, 14, 14), the main output is of size (batch_size, 2)
        :param auxiliary_loss: bool: Whether to use auxiliary losses, if true the model returns 2 predictions, output and class
        :param auxiliary_weight: float: The weight of the auxiliary loss compared to the main loss which has a weight of 1
        :return: if `auxiliary_loss`: returns a tuple of tensors which are (batch_size, 2), (batch_size, 10, 2).
        The first has predictions of whether the first digit is smaller than the latter, the second has the digit class prediction for each input channel.
        otherwise returns only the first tensor of size (batch_size,2)
        """
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
        """
        Forward pass of the model
        :param x: Tensor: input tensor
        :return: *Tensor, comparison predictions and, if self.auxiliary_loss, digit predictions
        """
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

