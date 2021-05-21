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
        raise NotImplementedError
       
    
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
            if len(p)!=0:#<--------------------------------
                u.append(empty(p[0][0].size()).fill_(0))
            else:
                u.append(None)
        self.u = u
        #print(self.u)
        
    def step(self):

        for module in self.model.module_list:
            for u,p in zip(self.u,module.param()):
                if not u is None:

                    print(p)
                    u = self.momentum * u
                    u += self.lr * p[1]
                    p[0] = p[0] - u
        '''for i, param_set in enumerate(self.model.param()):
            if not self.u[i] is None:
                for p in param_set:
                    print(p)
                    self.u[i] = self.momentum * self.u[i]
                    self.u[i] += self.lr * p[1]
                    p[0] = p[0] - self.u[i]'''