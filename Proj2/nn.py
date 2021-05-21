from torch import empty, Tensor
import math

class Module:
    """
    Base class for all other Modules
    """

    def __init__(self):
        super().__init__()

    def forward(self, *input_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        raise NotImplementedError
    
    
    
class Linear(Module):
    def __init__(self, in_features:int, out_features:int, bias=True):
        """
        Class of Linear module. Applies an affine transformation to the incoming data
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param bias: if set to False, the layer will not learn an additive bias. Default: True
        """
        super(Module, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = 1 / math.sqrt(in_features)
        self.w = empty(out_features, in_features)
        self.w.uniform_(-std, std)
        self.dl_dw = empty(out_features, in_features)
        self.dl_dw.zero_()
        self.bias = bias
        if bias:
            self.b = empty(out_features)
            self.b.uniform_(-std, std)
            self.dl_db = empty(out_features)
            self.dl_db.zero_()
    
    
    def forward(self, x:Tensor):
        """
        Perform forward pass
        :param x: input tensors
        :return: result of forward pass
        """
        self.x = x.clone()
        if self.bias:
            return x.mm(self.w.t()) + self.b
        #print(output)
        else:
            return x.mm(self.w.t())
        #print(output)

    
    
    def backward(self, dl_ds:Tensor):
        """
        Perform backward pass
        """
        # dl_db = dl_ds (sum over all samples in batch)
        # dl_dw = dl_ds @ x^(l-1).T (sum over samples in batch)
        # dl_dx = w.T @ dl_ds

        if self.bias:
            self.dl_db += dl_ds.sum(0)
        #print('X shape', self.x.shape)
        #print('dl_ds:', dl_ds.shape)
        self.dl_dw += dl_ds.t().mm(self.x)
        return dl_ds.mm(self.w)

    
    
    def param(self):
        """
        Parameters of module
        :return: a list of pairs, each composed of a parameter tensor and a gradient tensor with respect to the parameter tensor
        """
        if not self.bias:
            return [(self.w, self.dl_dw)]
        else:
            return [(self.w, self.dl_dw), (self.b, self.dl_db)]

    
    def zero_grad(self):
        self.dl_dw.fill_(0)
        if self.bias:
            self.dl_db.fill_(0)




class Sequential(Module):
    
    def __init__(self, *modules):
        super(Module, self).__init__()
        self.module_list = list(modules)

    def forward(self, input_):
        output = input_
        #print('FORWARD')
        for m in self.module_list:
            #print(output.shape)
            output = m.forward(output)
        return output

    def backward(self, gradwrtoutput):
        #print('BACK')
        grad = gradwrtoutput
        for i in range(len(self.module_list)-1, 0, -1):
            #print(type(self.module_list[i]))
            #print("gradient shape", grad.shape)
            grad = self.module_list[i].backward(grad)
        return grad

    def param(self):
        return [m.param() for m in self.module_list]

    def zero_grad(self):
        [m.zero_grad() for m in self.module_list]
    
    
    
class ReLU(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x:Tensor):
        self.grad = x.gt(0)
        return x.relu()

    def backward(self, dl_dx:Tensor):
        return self.grad*dl_dx

    def param(self):
        return []

    def zero_grad(self):
        pass
    
    
    
class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x:Tensor):
        self.x = x
        return x.tanh()

    def backward(self, dl_dx:Tensor):
        return dl_dx*(1-self.x.tanh().pow(2))

    def param(self):
        return []

    def zero_grad(self):
        pass
    
    
class LossMSE(Module):
    """
    Class of MSE loss
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred:Tensor, target:Tensor):
        """
        Perform forward pass
        :param pred: predicted value
        :param target: target value
        :return: result of forward pass
        """

        self.pred = pred
        self.target = target
        if len(target.size()) == 1:
            self.target = self.target.unsqueeze(1)
#         return (self.pred - self.target).pow(2).mean()
        return (self.pred - self.target).pow(2).mean()
    
    def backward(self):
        """
        Perform backward pass
        :return: gradient of the loss with respect to predicted values
        """
        #print('MSE')
        #print(self.pred.shape)
        #print(self.target.shape)
        return 2 * (self.pred - self.target) / (self.target.size(0))
#         return 2 * (self.pred - self.target) / (self.target.size()[0] * self.target.size()[1])