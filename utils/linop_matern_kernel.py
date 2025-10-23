import torch
import math
import gpytorch

class LinopMaternKernel(gpytorch.kernels.MaternKernel):
    def __init__(self, nu=1.5, lengthscale_prior=None, outputscale_prior=None, **kwargs):
        super().__init__(nu=nu, lengthscale_prior=lengthscale_prior, outputscale_prior=outputscale_prior, **kwargs)

    def _forward_no_kernel_linop(
            self, X1, X2
        ):
            X1_ = X1[..., :, None, :]
            X2_ = X2[..., None, :, :]
            diffs = X1_ - X2_
            if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
                dists = diffs.norm(dim=-1)
            else:
                dists = diffs.abs().squeeze(-1)
            consts = -math.sqrt(self.nu * 2)
            exp_component = (consts * dists).exp()

            if self.nu == 0.5:
                constant_component = torch.ones_like(dists)
            elif self.nu == 1.5:
                constant_component = torch.add(1.0, dists, alpha=math.sqrt(3))
            elif self.nu == 2.5:
                constant_component = torch.add(1.0, dists, alpha=math.sqrt(5)).add_(dists.square(), alpha=(5.0 / 3.0))

            return exp_component.mul(constant_component)