from torch import empty
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
    
    
    def forward(self, *inputs):
        """
        Perform forward pass
        :param x: input tensors
        :return: result of forward pass
        """
        output = []
        for x in inputs:
            #print(x)
            if not self.bias:
                output.append(x.mm(self.w.t()))
            else:
                output.append(x.mm(self.w.t()) + self.b)
        self.inputs = inputs
        #print(output)
        output = tuple(output)
        #print(output)
        return output
    
    
    def backward(self, *dl_ds):
        """
        Perform backward pass
        """
        # dl_db = dl_ds (sum over all samples in batch)
        # dl_dw = dl_ds @ x^(l-1).T (sum over samples in batch)
        # dl_dx = w.T @ dl_ds
        out = []
        for i,dl_dx_out in zip(self.inputs,dl_ds):
            if self.bias:
                self.dl_db.add(dl_dx_out.sum(0))
            self.dl_dw.add(dl_dx_out.view(-1,1).mm(i.sum(0).view(1,-1)))
            out.append(self.w.t().mm(dl_dx_out.view(-1, 1)).squeeze())
        return tuple(out)

    
    
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

    def forward(self, *input_):
        output = input_
        for m in self.module_list:
            #print(output)
            output = m.forward(*output)
        return output

    def backward(self, *gradwrtoutput):
        grad = gradwrtoutput
        for i in range(len(self.module_list)-1, 0, -1):
            print(grad[0].shape)
            grad = self.module_list[i].backward(*grad)
        return grad

    def param(self):
        return [m.param() for m in self.module_list]

    def zero_grad(self):
        [m.zero_grad() for m in self.module_list]
    
    
    
class ReLU(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *input_):
        for i in input_:
            #print(i)
            i[i<0] = 0
        self.inputs = input_
        return input_

    def backward(self, *gradwrtoutput):
        out = []
        for i,grad in zip(self.inputs,gradwrtoutput):
            out.append(i.gt(0)*grad)
        return tuple(out)

    def param(self):
        return []

    def zero_grad(self):
        pass
    
    
    
class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *input_):
        out = []
        for i in input_:
            out.append(i.tanh())
        self.inputs = input_
        return tuple(out)

    def backward(self, *gradwrtoutput):
        out = []
        for i,grad in zip(self.inputs, gradwrtoutput):
            out.append(grad * (1-i.tanh().pow(2)).sum(0))
        return tuple(out)

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
    
    def forward(self, pred, target):
        """
        Perform forward pass
        :param pred: predicted value
        :param target: target value
        :return: result of forward pass
        """
        self.pred = pred
        self.target = target
        return (self.pred - self.target).pow(2).mean()
    
    def backward(self):
        """
        Perform backward pass
        :return: gradient of the loss with respect to predicted values
        """
        return 2 * (self.pred - self.target) / (self.target.size()[0] * self.target.size()[1])