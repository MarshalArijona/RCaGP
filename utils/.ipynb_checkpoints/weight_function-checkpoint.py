import torch
import gpytorch
import numpy as np
from torch.nn import Module
from torch.nn.parameter import Parameter
from gpytorch.constraints import Positive

from abc import ABC, abstractmethod

def get_soft_threshold(Y, X, mean_module, epsilon=0.1):
    m = mean_module(X)
    abs_y_min_m = torch.absolute(Y - m)
    return torch.quantile(abs_y_min_m, epsilon)

class W(ABC):

    @abstractmethod
    def W(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dy(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        y.requires_grad_(True)
        w = self.W(X, y)
        grad_W_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        return grad_W_y
        
    def w_dy(self, X: torch.Tensor, y: torch.Tensor):
        return self.W(X, y), self.dy(X, y)

class StandardWeight(W):
    def __init__(self, noise_likelihood):
        self.noise = torch.nn.Parameter(noise_likelihood.detach())
        torch.set_default_dtype(torch.float64)

    def W(self, X:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        device = X.device
        n_data = y.shape[0]
        return torch.ones(n_data, device=device) * self.noise / torch.sqrt(torch.tensor(2))
    
    def dy(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device=X.device
        n_data = y.shape[0]
        return torch.zeros(n_data, device=device)

    def dylog2(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_data = y.shape[0]
        device = X.device
        return torch.zeros(n_data, device=device)

class IMQ(W):
    def __init__(self, 
        mean_module,
        beta=1.0,
        C=1.0) -> None:
        
        self.beta = torch.tensor(beta)
        self.C = torch.tensor(C)
        self.mean_module = mean_module
        torch.set_default_dtype(torch.float64)

    def W(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = self.mean_module(X)
        return self.beta * torch.sqrt(1/(1+( (y - m)**2 / self.C ** 2 )))

    def dy(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = self.mean_module(X)
        return self.beta* (- (y - m ) / (self.C**2 )) * torch.pow(1/(1+( ((y - m) ** 2) / (self.C ** 2 ))), 3/2)
   
    def dylog2(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = self.mean_module(X)
        result = - 2 * (y - m) / (self.C**2 + (y - m)**2 )
        return result