

class Module(object):
    """
    Base class for all other Modules
    """

    def __init__(self):
        pass

    def forward(self, *input_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass
    
    
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Class of Linear module. Applies a linear transformation to the incoming data
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :bias: if set to False, the layer will not learn an additive bias. Default: True
        """
        super(Module, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.w = torch.empty(out_features, in_features)
        self.dl_dw = torch.empty(out_features, in_features)
        
        if bias:
            self.b = torch.empty(out_features)
            self.dl_db = torch.empty(out_features)
        else:
            self.b = None
            self.dl_db = None
    
    
    def forward(self, x):
        """
        Perform forward pass
        :param x: input tensor
        :return: result of forward pass
        """
        self.x = x
        if self.b is None:
            output = self.x.mm(self.w.t())
        else
            output = self.x.mm(self.w.t()) + self.b
        return output
    
    
    def backward(self, dl_dx_out):
        """
        Perform backward pass
        """
    
    
    def param(self):
        """
        Parameters of module
        :return: a list of pairs, each composed of a parameter tensor and a gradient tensor with respect to the parameter tensor
        """
        if self.b is None:
            return [(self.w, self.dl_dw)]
        else:
            return [(self.w, self.dl_dw), (self.b, self.dl_db)]

    
    def zero_grad(self):
        pass



class Sequential(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *input_):
        pass

    def backward(self, *gradwrtoutput):
        pass

    def param(self):
        pass

    def zero_grad(self):
        pass
    
    
    
class ReLU(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *input_):
        pass

    def backward(self, *gradwrtoutput):
        pass

    def param(self):
        pass

    def zero_grad(self):
        pass
    
    
    
class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *input_):
        pass

    def backward(self, *gradwrtoutput):
        pass

    def param(self):
        pass

    def zero_grad(self):
        pass