import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import dlc_practical_prologue as prologue


def read_input(batch_size=100, single_channel=True, normalize=True, pairs=1000):
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

        train_loader = DataLoader(trainDataset, batch_size, shuffle=False)
        test_loader = DataLoader(testDataset, batch_size, shuffle=False)

        test_target_Dataset = TensorDataset(test_input, test_target)
        test_target_loader = DataLoader(test_target_Dataset, batch_size, shuffle=False)

        return train_loader, test_loader, test_target_loader

    else:
        trainDataset = TensorDataset(train_input, train_target)
        testDataset = TensorDataset(test_input, test_target)

        train_loader = DataLoader(trainDataset, batch_size, shuffle=False)
        test_loader = DataLoader(testDataset, batch_size, shuffle=False)

        return train_loader, test_loader


def train_model_epoch(model, optimizer, criterion, train_loader):
    tr_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

    tr_loss = tr_loss / (batch_idx + 1)
    return tr_loss


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


def train_model(model, train_loader, test_loader, learning_rate=1e-2, epochs=10, eval_=True):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr)
    # momentum=momentum

    tr_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        tr_loss = train_model_epoch(model, optimizer, criterion, train_loader)
        tr_loss_list.append(tr_loss)

        if (eval_):
            model.eval()
            val_loss, correct_pred = validate_model_epoch(model, criterion, test_loader)
            val_accuracy = correct_pred / (test_loader.batch_size * len(test_loader))
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy)

            print('\nEpoch: {}/{}, Train Loss: {:.6f}, Val Loss: {:.6f}, Val Accuracy: {:.6f}% {}/{}'.format(epoch + 1,
                                                                                                             epochs,
                                                                                                             tr_loss,
                                                                                                             val_loss,
                                                                                                             100 * val_accuracy,
                                                                                                             correct_pred,
                                                                                                             test_loader.batch_size * len(
                                                                                                                 test_loader)))
    #         else:
    #             print('\nEpoch: {}/{}, Train Loss: {:.6f}'.format(epoch + 1, epochs, tr_loss))

    return tr_loss_list, val_loss_list, val_accuracy_list

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
