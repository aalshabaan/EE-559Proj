import torch
from utils import *
from nn import *

# Disable autograd
torch.set_grad_enabled(False)

print('Project 2 Mini deep-learning framework running...')
print('\nNetwork architecture: 2 input units, 3 hidden layers of 25 units, 1 output unit, ReLU activations and MSE loss.')
print('Training over 15 epochs with SGD of 10 mini batch size and 0.1 learning rate.')

train_input, train_target = generate_disc_data()
test_input, test_target = generate_disc_data()
train_input, test_input = normalize(train_input, test_input)

model = Sequential(Linear(2, 25), ReLU(),
               Linear(25, 25), ReLU(),
               Linear(25, 25), ReLU(),
               Linear(25, 1))

print('\nLogging the loss:')
tr_loss_list, tr_acc_list = train_model(model, 
                                    train_input, 
                                    train_target,
                                    learning_rate=1e-1,
                                    epochs=15, 
                                    batch_size=10,
                                    optimizer='SGD', 
                                    loss='mse', 
                                    verbose=True, 
                                    momentum=0.0)

test_acc = evaluate_model(model, test_input, test_target, batch_size=10)
    

print('\nFinal accuracy: train={}, test={}'.format(tr_acc_list[-1], test_acc))