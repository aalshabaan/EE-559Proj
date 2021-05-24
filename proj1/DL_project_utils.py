import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import dlc_practical_prologue as prologue


def read_input(normalize=True, pairs=1000, verbose=False):
    """
    Read the data from prologue and return Tensor Datasets
    :param normalize: bool: True if we want to normalize the data
    :param pairs: int: The total number of pairs
    :param verbose: bool: If True print data information
    :return: trainDataset, testDataset: Torch TensorDataset objects to facilitate batch training
    """
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(pairs)
    
    if verbose:
        print('train_input', train_input.size(), 'train_target', train_target.size(), 'train_classes', train_classes.size())
        print('test_input', test_input.size(), 'test_target', test_target.size(), 'test_classes', test_classes.size())

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
        if verbose:
            print('normalize train and test datasets with train mean & std')

    trainDataset = TensorDataset(train_input, train_target, train_classes)
    testDataset = TensorDataset(test_input, test_target, test_classes)


    return trainDataset, testDataset


def train_model_epoch(model, optimizer, criterion, train_loader, cuda=False, auxiliary_loss=False, lbda=0.4):
    tr_loss = 0
    tr_acc = 0
    for batch_idx, (data, target, classes) in enumerate(train_loader):

        optimizer.zero_grad()
        if not auxiliary_loss:
            if cuda and torch.cuda.is_available():
                output = model(data.cuda())
                tr_acc += (output.max(1)[1] == target.cuda()).sum().item()

                loss = criterion(output, target.cuda())
                tr_loss += loss.item()
            else:
                output = model(data.cpu())
                tr_acc += (output.max(1)[1] == target.cpu()).sum().item()

                loss = criterion(output, target.cpu())
                tr_loss += loss.item()
        else:
            classes_0 = classes[:, 0]
            classes_1 = classes[:, 1]

            if cuda and torch.cuda.is_available():
                output, classes_output = model(data.cuda())
                tr_acc += (output.max(1)[1] == target.cuda()).sum().item()

                loss = criterion(output, target.cuda()) + lbda*criterion(classes_output[:,:,0], classes_0.cuda()) + lbda*criterion(classes_output[:,:,1], classes_1.cuda())
                tr_loss += loss.item()
            else:
                output, classes_output = model(data.cpu())
                tr_acc += (output.max(1)[1] == target.cpu()).sum().item()

                loss = criterion(output, target.cpu()) + lbda*criterion(classes_output[:,:,0], classes_0.cpu()) + lbda*criterion(classes_output[:,:,1], classes_1.cpu())
                tr_loss += loss.item()

        loss.backward()
        optimizer.step()

    tr_loss, tr_acc = tr_loss / (batch_idx + 1), (tr_acc)/len(train_loader.dataset)
    return tr_loss, tr_acc


def validate_model_epoch(model, criterion, test_loader, cuda=False, auxiliary_loss=False, lbda=0.4):
    val_loss = 0
    correct_pred = 0

    if auxiliary_loss:
        if cuda and torch.cuda.is_available():
            for batch_idx, (data, target, classes) in enumerate(test_loader):
                pred, class_pred = model(data.cuda())

                loss = criterion(pred, target.cuda()) + lbda*criterion(class_pred[:,:,0], classes[:,0].cuda()) + lbda*criterion(class_pred[:,:,1], classes[:,1].cuda())
                val_loss += loss.item()

                predicted_label = pred.max(1)[1]
                correct_pred += (predicted_label == target.cuda()).sum()
        else:
            for batch_idx, (data, target, classes) in enumerate(test_loader):
                pred, class_pred = model(data.cpu())

                loss = criterion(pred, target.cpu()) + lbda*criterion(class_pred[:,:,0], classes[:,0].cpu()) + lbda*criterion(class_pred[:,:,1], classes[:,1].cpu())
                val_loss += loss.item()

                predicted_label = pred.max(1)[1]
                correct_pred += (predicted_label == target.cpu()).sum()
    else:
        if cuda and torch.cuda.is_available():
            for batch_idx, (data, target, classes) in enumerate(test_loader):
                pred = model(data.cuda())

                loss = criterion(pred, target.cuda())
                val_loss += loss.item()

                predicted_label = pred.max(1)[1]
                correct_pred += (predicted_label == target.cuda()).sum()
        else:
            for batch_idx, (data, target, classes) in enumerate(test_loader):
                pred = model(data.cpu())

                loss = criterion(pred, target.cpu())
                val_loss += loss.item()

                predicted_label = pred.max(1)[1]
                correct_pred += (predicted_label == target.cpu()).sum()

    val_loss = val_loss / (batch_idx + 1)
    return val_loss, correct_pred


def train_model(model, train_dataset, learning_rate=1e-2, epochs=10, batch_size=100 , eval_=True, optimizer='SGD', loss='cross_entropy',
                validation_split=0.2, verbose=True, cuda=False, momentum=0.0, weight_decay=0.0):
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
    :param verbose: bool: If True print training statistics
    :param cuda: bool: If True move model to GPU
    :param momentum: float: momentum for SGD
    :param weight_decay: float: weight decay for ADAM
    :return: tr_loss_list, tr_acc_list, val_loss_list, val_accuracy_list: lists containing the training history
    """
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) #momentum=0.9
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #weight_decay=0.01 (L2 penalty)
    else:
        raise Exception('Unsupported optimizer type. Use SGD or Adam')

    if loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise Exception('Unsupported loss, use cross_entropy or mse')

    if eval_:
        validation_size = int(validation_split * len(train_dataset))
        train_size = len(train_dataset) - validation_size
        train_subset, validation_subset = torch.utils.data.random_split(dataset=train_dataset, lengths=[train_size, validation_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    if hasattr(model, 'auxiliary_loss'):
        auxiliary_loss = model.auxiliary_loss
        lbda = model.auxiliary_weight
    else:
        auxiliary_loss = False
        lbda = 0

    tr_loss_list = []
    tr_acc_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        tr_loss, tr_acc = train_model_epoch(model, optimizer, criterion, train_loader, cuda, auxiliary_loss, lbda)
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)

        if (eval_):
            model.eval()
            val_loss, correct_pred = validate_model_epoch(model, criterion, validation_loader, cuda, auxiliary_loss, lbda)
            val_accuracy = correct_pred / len(validation_loader.dataset)
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy.item())
        
        if verbose:
            if eval_:
                print('\nEpoch: {}/{}, Train Loss: {:.6f}, Train Accuracy {:.6f}% Val Loss: {:.6f}, Val Accuracy: {:.6f}%{}/{}'.format(epoch + 1, epochs, tr_loss, 100*tr_acc, val_loss, 100 * val_accuracy, correct_pred, len(validation_loader.dataset)))
            else:
                print('\nEpoch: {}/{}, Train Loss: {:.6f}, Train Accuracy {:.6f}%'.format(epoch + 1, epochs, tr_loss, 100*tr_acc))

    return tr_loss_list, tr_acc_list, val_loss_list, val_accuracy_list


def evaluate_model(model, test_data, cuda=False, batch_size=100):
    test_loader = DataLoader(test_data, batch_size=batch_size)

    if hasattr(model, 'auxiliary_loss'):
        auxiliary_loss = model.auxiliary_loss
    else:
        auxiliary_loss = False

    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        cuda = False
        model.cpu()
        
    model.eval()
    nb_errs = 0
    
    for data, target, _ in test_loader:
        if auxiliary_loss:
            if cuda:
                out, _ = model(data.cuda())
                nb_errs += (out.argmax(dim=1) == target.cuda()).sum().item()
            else:
                out, _ = model(data)
                nb_errs += (out.argmax(dim=1) == target).sum().item()
        else:
            if cuda:
                out = model(data.cuda())
                nb_errs += (out.argmax(dim=1) == target.cuda()).sum().item()
            else:
                out = model(data)
                nb_errs += (out.argmax(dim=1) == target).sum().item()

    return nb_errs/len(test_data)