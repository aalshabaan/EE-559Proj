from torch import empty

class Optim(object):
    """
	Base class for optimization algorithms
    """
    def step(self, *input):
        """
        Performs a single optimization step. Parameter update
        :param input: parameters for step function
        """
        raise NonImplementedError
       
    
class SGD(Optim):
    def __init__(self, model, lr, momentum=0):
        """
        Stochastic Gradient Descent optimization algorithm (optionally with momentum)
        :param model: model to optimize
        :param lr: learning rate
        :param momentum: momentum factor (default: 0)
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        u = []
        for p in self.model.param():
            u.append(torch.empty(p[0].size()).fill_(0))
        self.u = u
        
    def step(self):
        for i, p in enumerate(self.model.param()):
            self.u[i] = self.momentum * self.u[i] + self.lr * p[1]
            p[0] = p[0] - self.u[i]