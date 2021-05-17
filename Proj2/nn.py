

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