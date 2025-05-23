# Parent class for differnet objectives/tasks
import numpy as np
import torch

class Objective:
    """Base class for any optimization task
    class supports oracle calls and tracks
    the total number of oracle class made during
    optimization
    """

    def __init__(
        self,
        dim,
        num_calls=0,
        dtype=torch.float32,
        lb=None,
        ub=None,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim 
        self.dim = dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype

    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), each item in enumerable type must be a float tensor of shape (dim,) (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by passing each x in xs into f(x).

        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        ys = []
        for x in xs:
            ys.append(self.f(x))
        return torch.tensor(ys).to(dtype=self.dtype).unsqueeze(-1)

    def inject_contamination(self, ys, std, noise, mask, low=1.5, high=2.0):
        y_contaminated = ys.reshape(-1, 1) + mask.reshape(-1, 1) * (low * std + ((high - low) * std) * noise).reshape(-1, 1) 
        return y_contaminated

    def asymmetric_contamination(self, ys, seed=0, probability=0.25):
        n_data = ys.shape[0]
        mask = torch.bernoulli(torch.full((n_data,), probability))
        outliers_idx = torch.nonzero(mask, as_tuple=True)[0]
        torch.manual_seed(seed)
        uniform_noises = torch.rand(n_data)        
        return uniform_noises, outliers_idx, mask

    def f(self, x):
        """
        Function f defines function f(x) we want to optimize. This method should also increment self.num_calls by one.

        Args:
            x (tensor): (1, dim) float tensor giving vector in input search space.

        Returns:
            float: reward value obtained by evaluating f at x
        """
        raise NotImplementedError(
            "Must implement f() specific to desired optimization task"
        )