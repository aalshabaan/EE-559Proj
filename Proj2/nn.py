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
        :return: result of forward pass, an affine transformation of the input
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
        Perform a back pass on the linear layer, accumulating the loss gradient with respect to the parameters
        :param dl_ds: Tensor, the loss gradient with respect to the layer's output
        :return: Tensor, the loss gradient with respect to the layer's input
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
        """
        Resets the module's accumulated gradient to zero, preparing for a new round of updates
        """
        self.dl_dw.fill_(0)
        if self.bias:
            self.dl_db.fill_(0)




class Sequential(Module):
    
    def __init__(self, *modules: Module):
        """
        Module allowing the combination of multiple other modules in a simple sequential architecture
        :param modules: The modules to be combined into the model ordered from first to last, that's to say input to output
        """
        super(Module, self).__init__()
        self.module_list = list(modules)

    def forward(self, input_: Tensor):
        """
        Performs a forward pass of the sequential model by doing forward passes of all the encompassed modules
        :param input_: Tensor: The input tensor to the model
        :return: The output of the model
        """
        output = input_
        for m in self.module_list:
            output = m.forward(output)
        return output

    def backward(self, gradwrtoutput: Tensor):
        """
        Performs a backward pass of the model by performing backward passes on all the modules from end to start
        :param gradwrtoutput: Tensor: The loss gradient with respect to the model's output
        :return: Tensor: The loss gradient with respect to the model's input
        """
        grad = gradwrtoutput
        for i in range(len(self.module_list)-1, 0, -1):
            grad = self.module_list[i].backward(grad)
        return grad

    def param(self):
        """
        Parameters of all the model's module parameters
        :return: List of lists of pairs corresponding to the model's modules' parameters and gradients
        """
        return [m.param() for m in self.module_list]

    def zero_grad(self):
        """
        Resets the gradients of all the model's modules
        """
        [m.zero_grad() for m in self.module_list]
    
    
    
class ReLU(Module):
    
    def __init__(self):
        """
        Module representing a ReLU activation function, keeps the positive input unchanged and sets the output 0 for negative input
        """
        super().__init__()

    def forward(self, x:Tensor):
        """
        Forward pass of the ReLU module. Keeps the positive input unchanged and sets the output 0 for negative input
        :param x: Tensor: input tensor
        :return: Tensor, output tensor
        """
        self.grad = x.gt(0) # No need to keep track of the input value, only if it's positive or negative
        return x.relu()

    def backward(self, dl_dx:Tensor):
        """
        Backward pass of the ReLU module, sets the gradient to 0 where ReLU's input is negative
        :param dl_dx: Tensor: loss gradient with respect to the output
        :return: Tensor: loss gradient with respect to the input
        """
        return self.grad*dl_dx

    def param(self):
        """
        returns an empty list, here to facilitate the inclusion of ReLU in a sequential model
        :return: empty list
        """
        return []

    def zero_grad(self):
        pass
    
    
    
class Tanh(Module):
    
    def __init__(self):
        """
        Module representing the tanh output function
        """
        super().__init__()
        
    def forward(self, x:Tensor):
        """
        Pass the input through a tanh function component-wise
        :param x: Tensor: Input tensor
        :return: Tensor: output tensor
        """
        self.x = x # We keep track of the input to calculate the module's gradient
        return x.tanh()

    def backward(self, dl_dx:Tensor):
        """
        Backward pass of the tanh module
        :param dl_dx: Tensor: loss gradient with respect to the output
        :return: Tensor: loss gradient with respect to the input
        """
        return dl_dx*(1-self.x.tanh().pow(2))

    def param(self):
        """
        returns an empty list, here to facilitate the inclusion of tanh in a sequential model
        :return: empty list
        """
        return []

    def zero_grad(self):
        pass
    
    
class LossMSE(Module):

    def __init__(self):
        """
        Module representing the MSE Loss
        """
        super().__init__()
    
    def forward(self, pred:Tensor, target:Tensor):
        """
        Perform forward pass of MSE Loss
        :param pred: predicted value
        :param target: target value
        :return: result of forward pass
        """

        self.pred = pred
        self.target = target
        if len(target.size()) == 1:
            self.target = self.target.unsqueeze(1)
        return (self.pred - self.target).pow(2).mean()
    
    def backward(self):
        """
        Perform backward pass
        :return: gradient of the loss with respect to the predicted values
        """
        return 2 * (self.pred - self.target) / (self.target.size(0))
