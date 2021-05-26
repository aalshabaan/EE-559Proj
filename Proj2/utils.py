"""
Defines methods for creating the input and training-evaluating the models.
"""
import torch
import math
from optim import SGD
from nn import LossMSE


def generate_disc_data(nb_points=1000):
    """
    Generates a set of 1000 2d points sampled uniformly in [0, 1], each with a label 0
    if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside.
    :param nb_points: int: Number of samples
    :return: torch.FloatTensor, torch.IntTensor: Input and Target data
    """
    input_ = torch.empty(nb_points, 2).uniform_(0, 1)
    target_ = ((input_-0.5).pow(2).sum(dim=1)<=1/(2*math.pi)).int()
    return input_, target_


def normalize(train_input, test_input):
    """
    Normalizes the train and test sets based on train data mean and std.
    :param train_input: torch.FloatTensor: Train data
    :param test_input: torch.FloatTensor: Test data
    :return: torch.FloatTensor, torch.FloatTensor: Normalized train and test data
    """
    mean, std = train_input.mean(dim=0), train_input.std(dim=0)
    train_norm, test_norm = (train_input-mean)/std, (test_input-mean)/std
    return train_norm, test_norm


def train_model_epoch(model, train_input, train_target, optimizer, criterion, batch_size):
    """
    This function trains a nn.Sequential model with the training dataset according to the specified loss
    and optimizer modules for one epoch and returns the loss and accuracy.
    :param model: nn.Sequential: The model to train for one epoch
    :param train_input: torch.FloatTensor: The train input dataset
    :param train_target: torch.IntTensor: The train target dataset
    :param optimizer: optim.SGD: Optimizer module
    :param criterion: nn.LossMSE: MSE loss module
    :param batch_size: int: The batch size
    :return: tr_loss, tr_acc: Training loss and accuracy for one epoch or training
    """
    tr_loss = 0
    tr_acc = 0
    data_size = train_input.size()[0]
    
    for data,target in zip(train_input.split(batch_size), train_target.split(batch_size)):

        model.zero_grad()
        output = model.forward(data)
        
        loss = criterion.forward(output, target)
        
        tr_loss += loss
        tr_acc += (output.gt(0.5).squeeze() == target.squeeze()).sum().item()


        l_grad = criterion.backward()
        model.backward(l_grad)
        optimizer.step()

    tr_loss, tr_acc = tr_loss/data_size, tr_acc/data_size
    return tr_loss, tr_acc


def train_model(model, train_input, train_target, learning_rate=1e-2, epochs=10,
                batch_size=100, optimizer='SGD', loss='mse', verbose=True, momentum=0.0):
    """
    This function trains a nn.Sequential model with the training dataset according to
    the specified parameters for many epochs. During the training we keep track of the loss and accuracy.
    :param model: nn.Sequential: The model to train
    :param train_input: torch.FloatTensor: The train input dataset
    :param train_target: torch.IntTensor: The train target dataset
    :param learning_rate: float: The learning rate
    :param epochs: int: The maximum number of epochs for training
    :param batch_size: int: The batch size
    :param optimizer: str: 'SGD' optimizer is supported
    :param loss: str: 'mse' loss is supported
    :param verbose: bool: If True print training statistics
    :param momentum: float: Momentum for SGD
    :return: tr_loss_list, tr_acc_list: Lists containing the training loss and accuracy history
    """
    
    if optimizer == 'SGD':
        optimizer = SGD(model, lr=learning_rate, momentum=momentum)
    else:
        raise Exception('Unsupported optimizer type, use "SGD".')

    if loss == 'mse':
        criterion = LossMSE()
    else:
        raise Exception('Unsupported loss, use "mse".')
    
    tr_loss_list = []
    tr_acc_list = []  
    
    for epoch in range(epochs):
        tr_loss, tr_acc = train_model_epoch(model, train_input, train_target, optimizer, criterion, batch_size)
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)
        
        if verbose:
            print('\nEpoch: {}/{}, Train Loss: {:.6f}, Train Accuracy {:.6f}%'.format(epoch + 1, epochs, tr_loss, 100*tr_acc))

    return tr_loss_list, tr_acc_list


def evaluate_model(model, test_input, test_target, batch_size=100):
    """
    This function evaluates a trained nn.Sequential model with the test dataset per batches and reports the test accuracy.
    :param model: nn.Sequential: The trained model
    :param test_input: torch.FloatTensor: The test input dataset
    :param test_target: torch.IntTensor: The test target dataset
    :param batch_size: int: The batch size
    :return: float: Test accuracy
    """
    test_acc = 0
    
    for data,target in zip(test_input.split(batch_size), test_target.split(batch_size)):

        output = model.forward(data)
        test_acc += (output.gt(0.5).squeeze() == target.squeeze()).sum().item()

    return  test_acc/test_input.size()[0]