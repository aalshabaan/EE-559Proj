from DL_project_utils import train_model, evaluate_model, read_input
import matplotlib.pyplot as plt
from models import *
import numpy as np


results_mean = {}
results_std = {}

nb_rounds = 10
test_acc = np.empty(nb_rounds)

print('BASELINE running...')
lr, bs, opt= 0.001, 60, "Adam"
k1, k2 = 3, 3
c1, c2 = 32, 32
h1, h2 = 128, 10
dropout = 0.2
epochs = 25
model = CNN_1(batchNorm = True, double_conv=True, nb_channels_1=c1, nb_channels_2=c2, kernel_1=k1, kernel_2=k2, nb_hidden_1=h1, nb_hidden_2=h2, dropout_p=dropout)

for i in range(nb_rounds):
    train_data, test_data = read_input()
    _, _, _, _ = train_model(model,
                                train_dataset=train_data,
                                learning_rate=lr,
                                epochs=epochs,
                                batch_size=bs,
                                eval_=False,
                                optimizer=opt,
                                loss='cross_entropy',
                                verbose=False,
                                cuda=True)
    test_acc[i] = evaluate_model(model, test_data, batch_size=bs, cuda=True)
results_mean['Baseline'] = test_acc.mean()
results_std['Baseline'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['Baseline'],results_std['Baseline'],nb_rounds))

 
print('RESNET running...')
epochs = 25
optimizer = 'Adam'
lr = 1e-2
model = ResNet(nb_residual_blocks = 10, nb_channels = 10, kernel_size = 5,
               nb_classes = 10, nb_pred = 2,
               auxiliary_loss=False)
for i in range(nb_rounds):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, eval_=False, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['resnet'] = test_acc.mean()
results_std['resnet'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['resnet'],results_std['resnet'],nb_rounds))


print('SIAMESE running...')
epochs = 25
optimizer = 'Adam'
lr = 1e-2
model = SiameseNet1(auxiliary_loss=False)
for i in range(nb_rounds):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, eval_=False, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing'] = test_acc.mean()
results_std['weight_sharing'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['weight_sharing'],results_std['weight_sharing'],nb_rounds))


print('RESNET_AUX running...')
epochs = 25
optimizer = 'Adam'
lr = 1e-2
model = ResNet(nb_residual_blocks = 10, nb_channels = 10, kernel_size = 5,
               nb_classes = 10, nb_pred = 2,
               auxiliary_loss=True, auxiliary_weight=0.4)
for i in range(nb_rounds):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, eval_=False, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['resnet_aux'] = test_acc.mean()
results_std['resnet_aux'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['resnet_aux'],results_std['resnet_aux'],nb_rounds))


print('SIAMESE_AUX_FULL running...')
epochs = 25
optimizer = 'Adam'
lr = 1e-2
model = SiameseNet1(auxiliary_loss=True, auxiliary_weight=0.4)
for i in range(nb_rounds):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, eval_=False, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing_aux'] = test_acc.mean()
results_std['weight_sharing_aux'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['weight_sharing_aux'],results_std['weight_sharing_aux'],nb_rounds))



print('SIAMESE_AUX_HALF running...')
epochs = 25
optimizer = 'Adam'
lr = 1e-2
model = SiameseNet2(auxiliary_loss=False, auxiliary_weight=0.4)
for i in range(nb_rounds):

    train_data, test_data = read_input()
    _,_,_,_ = train_model(model, train_data, epochs=epochs, eval_=False, cuda=True, verbose=False, optimizer=optimizer, learning_rate=lr)
    test_acc[i] = evaluate_model(model, test_data, cuda=True)
results_mean['weight_sharing_half_aux'] = test_acc.mean()
results_std['weight_sharing_half_aux'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['weight_sharing_half_aux'],results_std['weight_sharing_half_aux'],nb_rounds))


with open('results.txt', 'w') as f:
    f.write('# Means\n')
    f.write(str(results_mean)+'\n')
    f.write('# Standard Deviations\n')
    f.write(str(results_std))