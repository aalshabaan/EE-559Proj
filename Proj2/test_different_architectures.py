import torch
import math
from utils import *
from nn import *
import numpy as np

# Disable autograd
torch.set_grad_enabled(False)

results_mean = {}
results_std = {}

nb_rounds = 10
test_acc = np.empty(nb_rounds)



print('Using ReLU activations...')
for i in range(nb_rounds):
    
    train_input, train_target = generate_disc_data()
    test_input, test_target = generate_disc_data()
    train_input, test_input = normalize(train_input, test_input)
    
    model = Sequential(Linear(2, 25), ReLU(),
                   Linear(25, 25), ReLU(),
                   Linear(25, 25), ReLU(),
                   Linear(25, 1))
    
    tr_loss_list, tr_acc_list = train_model(model, 
                                        train_input, 
                                        train_target,
                                        learning_rate=1e-1,
                                        epochs=25, 
                                        batch_size=10,
                                        optimizer='SGD', 
                                        loss='mse', 
                                        verbose=False, 
                                        momentum=0.0)
    
    test_acc[i] = evaluate_model(model, test_input, test_target, batch_size=10)
    
results_mean['baseline_ReLU'] = test_acc.mean()
results_std['baseline_ReLU'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['baseline_ReLU'],results_std['baseline_ReLU'],nb_rounds))



print('Using Tanh activations...')
for i in range(nb_rounds):
    
    train_input, train_target = generate_disc_data()
    test_input, test_target = generate_disc_data()
    train_input, test_input = normalize(train_input, test_input)
    
    model = Sequential(Linear(2, 25), Tanh(),
                   Linear(25, 25), Tanh(),
                   Linear(25, 25), Tanh(),
                   Linear(25, 1))
    
    tr_loss_list, tr_acc_list = train_model(model, 
                                        train_input, 
                                        train_target,
                                        learning_rate=1e-1,
                                        epochs=25, 
                                        batch_size=10,
                                        optimizer='SGD', 
                                        loss='mse', 
                                        verbose=False, 
                                        momentum=0.0)
    
    test_acc[i] = evaluate_model(model, test_input, test_target, batch_size=10)
    
results_mean['baseline_Tanh'] = test_acc.mean()
results_std['baseline_Tanh'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['baseline_Tanh'],results_std['baseline_Tanh'],nb_rounds))



print('Using ReLU activations and Tanh for output layer...')
for i in range(nb_rounds):
    
    train_input, train_target = generate_disc_data()
    test_input, test_target = generate_disc_data()
    train_input, test_input = normalize(train_input, test_input)
    
    model = Sequential(Linear(2, 25), ReLU(),
                   Linear(25, 25), ReLU(),
                   Linear(25, 25), ReLU(),
                   Linear(25, 1), Tanh())
    
    tr_loss_list, tr_acc_list = train_model(model, 
                                        train_input, 
                                        train_target,
                                        learning_rate=1e-1,
                                        epochs=25, 
                                        batch_size=10,
                                        optimizer='SGD', 
                                        loss='mse', 
                                        verbose=False, 
                                        momentum=0.0)
    
    test_acc[i] = evaluate_model(model, test_input, test_target, batch_size=10)
    
results_mean['baseline_ReLU_Tanh'] = test_acc.mean()
results_std['baseline_ReLU_Tanh'] = test_acc.std()
print('mean={}, std={} with {} rounds'.format(results_mean['baseline_ReLU_Tanh'],results_std['baseline_ReLU_Tanh'],nb_rounds))


with open('results.txt', 'w') as f:
    f.write('# Means\n')
    f.write(str(results_mean)+'\n')
    f.write('# Standard Deviations\n')
    f.write(str(results_std))