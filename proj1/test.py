from DL_project_utils import train_model, evaluate_model, read_input
import matplotlib.pyplot as plt
from models import *
import numpy as np
from tqdm import tqdm

results_mean = {}
results_std = {}

nb_rounds = 10
epochs = 25
optimizer = 'Adam'
lr = 1e-2


test_acc = np.empty(nb_rounds)
model = Net3()
print('BASELINE')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['Baseline'] = test_acc.mean()
results_std['Baseline'] = test_acc.std()

model = ResNet(nb_residual_blocks = 10, nb_channels = 10, kernel_size = 5,
               nb_classes = 10, nb_pred = 2,
               auxiliary_loss=False)
print('RESNET')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['resnet'] = test_acc.mean()
results_std['resnet'] = test_acc.std()

model = SiameseNet1(auxiliary_loss=False)
print('SIAMESE')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing'] = test_acc.mean()
results_std['weight_sharing'] = test_acc.std()

model = ResNet(nb_residual_blocks = 10, nb_channels = 10, kernel_size = 5,
               nb_classes = 10, nb_pred = 2,
               auxiliary_loss=True, auxiliary_weight=0.4)
print('RESNET_AUX')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['resnet_aux'] = test_acc.mean()
results_std['resnet_aux'] = test_acc.std()

model = SiameseNet1(auxiliary_loss=True, auxiliary_weight=0.4)
print('SIAMESE_AUX_FULL')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing_aux'] = test_acc.mean()
results_std['weight_sharing_aux'] = test_acc.std()

model = SiameseNet2(auxiliary_loss=False, auxiliary_weight=0.4)
print('SIAMESE_AUX_HALF')
for i in tqdm(range(nb_rounds)):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing_half_aux'] = test_acc.mean()
results_std['weight_sharing_half_aux'] = test_acc.std()


with open('results.txt', 'w') as f:
    f.write('# Means\n')
    f.write(str(results_mean)+'\n')
    f.write('# Standard Deviations\n')
    f.write(str(results_std))

