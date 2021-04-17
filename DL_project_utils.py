import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import dlc_practical_prologue as prologue


def read_input(batch_size=100, single_channel=True, normalize=True, pairs=1000, validation_split=None):
    """
    Read the data from prologue and return data loaders
    :param batch_size: int: the batch size
    :param single_channel: bool: True if we want both images concatenated
    :param normalize: bool: True if we want to normalize the data
    :param pairs: int: The total number of pairs, note that if single_channel then the returned pictures are twice this number
    :param validation_split: float: The portion of the training set to be used as validation data
    :return: train_loader, test_loader: Torch DataLoader objects to facilitate batch training
    """
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(pairs)

    #     print('train_input', train_input.size(), 'train_target', train_target.size(), 'train_classes', train_classes.size())
    #     print('test_input', test_input.size(), 'test_target', test_target.size(), 'test_classes', test_classes.size())

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
    #         print('normalize: ', normalize)

    if single_channel:
        train_input_sc = torch.cat((train_input[:, 0, :, :], train_input[:, 1, :, :]), 0).unsqueeze(1)
        train_classes_sc = torch.cat((train_classes[:, 0], train_classes[:, 1]), 0)
        test_input_sc = torch.cat((test_input[:, 0, :, :], test_input[:, 1, :, :]), 0).unsqueeze(1)
        test_classes_sc = torch.cat((test_classes[:, 0], test_classes[:, 1]), 0)


        #         print('single channel: train_input_sc', train_input_sc.size(), 'train_target', train_target.size(), 'train_classes_sc', train_classes_sc.size())
        #         print('single channel: test_input_sc', test_input_sc.size(), 'test_target', test_target.size(), 'test_classes_sc', test_classes_sc.size())

        trainDataset = TensorDataset(train_input_sc, train_classes_sc)
        testDataset = TensorDataset(test_input_sc, test_classes_sc)


        #train_loader = DataLoader(trainDataset, batch_size, shuffle=False)
        #test_loader = DataLoader(testDataset, batch_size, shuffle=False)

        test_target_Dataset = TensorDataset(test_input, test_target)
        #test_target_loader = DataLoader(test_target_Dataset, batch_size, shuffle=False)

        return trainDataset, testDataset, test_target_Dataset

    else:
        trainDataset = TensorDataset(train_input, train_target)
        testDataset = TensorDataset(test_input, test_target)

        #train_loader = DataLoader(trainDataset, batch_size, shuffle=False)
        #test_loader = DataLoader(testDataset, batch_size, shuffle=False)

        return trainDataset, testDataset


def train_model_epoch(model, optimizer, criterion, train_loader):
    tr_loss = 0
    tr_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        tr_acc += (output.max(1)[1] == target).sum()

        loss = criterion(output, target)
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

    tr_loss, tr_acc = tr_loss / (batch_idx + 1), (tr_acc)/len(train_loader.dataset)
    return tr_loss, tr_acc


def validate_model_epoch(model, criterion, test_loader):
    val_loss = 0
    correct_pred = 0


    for batch_idx, (data, target) in enumerate(test_loader):
        pred = model(data)

        loss = criterion(pred, target)
        val_loss += loss.item()

        predicted_label = pred.max(1)[1]
        correct_pred += (predicted_label == target).sum()

    val_loss = val_loss / (batch_idx + 1)
    return val_loss, correct_pred


def train_model(model, train_dataset, learning_rate=1e-2, epochs=10, batch_size=100 , eval_=True, optimizer='SGD', loss='cross_entropy',
                validation_split=0.2, verbose=True):
    """
    Trains the passed PyTorch model on the passed training dataset, the validation data is fixed and isn't randomly sampled at each epoch
    :param model: nn.Module: The model to train
    :param train_dataset: torch.util.data.TensorDataset: The training dataset
    :param learning_rate: float: The learning rate
    :param epochs: int: The maximum number of epochs for training
    :param batch_size: int: The batch size
    :param eval_: bool: True if we want to validate
    :param optimizer: str: 'SGD' or 'ADAM, which optimizer to use
    :param loss: str: 'mse' or 'cross_entropy', which loss to use
    :param validation_split: float: 0.0 to 1.0, the fraction of the dataset to be used for validation. Has no effect if eval_==False
    :return: tr_loss_list, val_loss_list, val_accuracy_list: lists containing the training history
    """
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise Exception('Unsupported optimizer type. Use SGD or Adam')

    if loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise Exception('Unsupported loss, use cross_entropy or mse')

    # momentum=momentum
    if eval_:
        validation_size = int(validation_split * len(train_dataset))
        train_size = len(train_dataset) - validation_size
        train_subset, validation_subset = torch.utils.data.random_split(dataset=train_dataset, lengths=[train_size, validation_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    tr_loss_list = []
    tr_acc_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        tr_loss, tr_acc = train_model_epoch(model, optimizer, criterion, train_loader)
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)

        if (eval_):
            model.eval()
            val_loss, correct_pred = validate_model_epoch(model, criterion, validation_loader)
            val_accuracy = correct_pred / (validation_loader.batch_size * len(validation_loader))
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy)
            if verbose:
                print('\nEpoch: {}/{}, Train Loss: {:.6f}, Train Accuracy {:.6f}% Val Loss: {:.6f}, Val Accuracy: {:.6f}% {}/{}'.format(epoch + 1,
                                                                                                                 epochs,
                                                                                                                 tr_loss,
                                                                                                                 tr_acc,
                                                                                                                 val_loss,
                                                                                                                 100 * val_accuracy,
                                                                                                                 correct_pred,
                                                                                                                 validation_loader.batch_size * len(
                                                                                                                     validation_loader)))
    #         else:
    #             print('\nEpoch: {}/{}, Train Loss: {:.6f}'.format(epoch + 1, epochs, tr_loss))

    return tr_loss_list, tr_acc_list, val_loss_list, val_accuracy_list

def predict_sc_accuracy_round(model, test_loader):
    model.eval()

    correct_pred = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        pred_first = model(data[:,0,:,:].unsqueeze(1))
        predicted_label_first = pred_first.max(1)[1]

        pred_second = model(data[:,1,:,:].unsqueeze(1))
        predicted_label_second = pred_second.max(1)[1]

        predicted_label = (predicted_label_first <= predicted_label_second)
        correct_pred += (predicted_label.int() == target).sum()

    return correct_pred


def predict_sc_accuracy(model, batch_size, learning_rate, epochs, rounds):
    val_accuracy_list = []

    for round_ in range(rounds):
        train_loader, test_loader, test_target_loader = read_input(batch_size)


        _, _, _ = train_model(model, train_loader, test_loader, learning_rate, epochs, eval_=False)

        nb_correct_pred = predict_sc_accuracy_round(model, test_target_loader)

        val_accuracy = nb_correct_pred / (test_target_loader.batch_size * len(test_target_loader))
        val_accuracy_list.append(val_accuracy)

        print('\nRound: {}/{}, Val Accuracy: {:.6f}% {}/{}'.format(round_ + 1, rounds, 100 * val_accuracy,
                                                                   nb_correct_pred, test_target_loader.batch_size * len(
                test_target_loader)))

    return val_accuracy_list
