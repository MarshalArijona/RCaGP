import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel


class LinopRBFKernel(RBFKernel):
    def __init__(self, lengthscale_prior=None, outputscale_prior=None, **kwargs):
        super().__init__(lengthscale_prior=lengthscale_prior, **kwargs)

    def _forward_no_kernel_linop(self, X1, X2):
        # Compute pairwise Euclidean distance
        X1_ = X1[..., :, None, :]
        X2_ = X2[..., None, :, :]
        diffs = X1_ - X2_

        if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
            dists = diffs.norm(dim=-1)
        else:
            dists = diffs.abs().squeeze(-1)
        
        dists_squared = dists.square()

        return (-0.5 * dists_squared).exp()
