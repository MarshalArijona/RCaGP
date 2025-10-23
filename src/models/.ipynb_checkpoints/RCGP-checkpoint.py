import torch
import gpytorch.settings as settings
import gpytorch.kernels as kernels

import sys 
sys.path.append("../")

from utils.cholesky_solve import cholesky_solve
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior

class RCGP(ExactGP):
    def __init__(self, 
            train_x, 
            train_y,
            mean_module,
            covar_module,
            likelihood,
            weight_function):
        
        super().__init__(
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=likelihood,
        )

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.weight_function = weight_function
        self.num_outputs = 1
        self.device = train_x.device

    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        
        predictive_mean, predictive_covar =  self._batch_call(x)
        dist = MultivariateNormal(predictive_mean, predictive_covar)

        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        else:
            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop

            f_mean = self.mean_module(self.train_inputs[0])
            jitter = 1e-6
            n_data = self.train_inputs[0].shape[0]
            f_covar = kernel_forward_fn(
                    self.train_inputs[0].div(lengthscale),
                    self.train_inputs[0].div(lengthscale),
                ).mul(outputscale) + torch.eye(n_data, device=self.device) * jitter

            # Compute J_w and y - mw
            w = self.weight_function.W(self.train_inputs[0], self.train_targets)
            sigma_sq = self.likelihood.noise
            J_w = torch.diag(sigma_sq / 2  * 1 / w ** 2)

            #Solve for the inverse of the gram matrix
            chol_factor = torch.linalg.cholesky(f_covar + sigma_sq * J_w)
            K_inv = cholesky_solve(torch.eye(chol_factor.size(-1), device=self.device), chol_factor)

            m_w = f_mean + self.weight_function.dylog2(self.train_inputs[0], self.train_targets)
            y_min_mw = self.train_targets - m_w
            k_x = kernel_forward_fn(
                    x
                    .div(lengthscale),
                    self.train_inputs[0]
                    .div(lengthscale)
            ).mul(outputscale)

            predictive_mean = self.mean_module(x) + k_x  @ K_inv @ y_min_mw
            predictive_covar = self.covar_module(x).add_jitter() - k_x @ K_inv @ k_x.T

            return predictive_mean, predictive_covar
        
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:

        self.eval()  
        self.likelihood.eval()
        fn = self(X)
        dist = self.likelihood(fn)

        return GPyTorchPosterior(dist)