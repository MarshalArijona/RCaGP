import torch
import gpytorch
import gpytorch.settings as settings
import gpytorch.kernels as kernels
from gpytorch.likelihoods import Likelihood

import sys 
sys.path.append("../")

from utils.cholesky_solve import cholesky_solve
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import GreaterThan
#from gpytorch.lazy import DiagLazyTensor
        
class RRP(ExactGP):
    def __init__(self, 
            train_x, 
            train_y,
            inducing_points,
            mean_module,
            covar_module,
            likelihood,
            learn_inducing_locations,):
        
        super().__init__(
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=likelihood,
        )

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.device = train_x.device

        self.S = []
        self.rho = []

        for i in range(len(self.train_targets)):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.rho.append(likelihood)

        self.num_outputs = 1
        self.learn_inducing_locations = learn_inducing_locations
        if self.learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points.clone())
        else:
            self.inducing_points = inducing_points.clone()
        self.num_inducing = self.inducing_points.size(-2)
        

    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        
        predictive_mean, predictive_covar =  self._batch_call(x)
        dist = MultivariateNormal(predictive_mean, predictive_covar)

        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        
        if self.training:
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            mean = self.mean_module(full_inputs)
            covar = self.covar_module(full_inputs).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            mean = self.mean_module(full_inputs)
            covar = self.covar_module(full_inputs).add_jitter()
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
            z_mean = self.mean_module(self.inducing_points)
            sigma_sq = self.likelihood.noise
            jitter=1e-6
            
            z_size = self.inducing_points.size(0)
            K_zz = kernel_forward_fn(
                self.inducing_points.div(lengthscale),
                self.inducing_points.div(lengthscale),
            ).mul(outputscale) + jitter * torch.eye(z_size, device=self.device)

            chol_K_zz = torch.linalg.cholesky(K_zz)
            K_zz_inv = cholesky_solve(torch.eye(chol_K_zz.size(-1), device=self.device), chol_K_zz) 

            K_xz = kernel_forward_fn(
                    x
                    .div(lengthscale),
                    self.inducing_points
                    .div(lengthscale)
            ).mul(outputscale)

            K_nz = kernel_forward_fn(
                self.train_inputs[0]
                .div(lengthscale),
                self.inducing_points
                .div(lengthscale)
            ).mul(outputscale)
            
            D = torch.zeros(len(self.train_targets), device=self.device)
            for idx in range(len(self.rho)):
                rho_i = self.rho[idx].noise
                D[idx] = rho_i

            rho_inv_diag = torch.diag(1.0 / (D + sigma_sq))

            Sigma_inv = K_zz +  K_nz.T @ rho_inv_diag @ K_nz
            chol_Sigma_inv = torch.linalg.cholesky(Sigma_inv)
            Sigma = cholesky_solve(torch.eye(chol_Sigma_inv.size(-1), device=self.device), chol_Sigma_inv)
            bmu = z_mean + K_zz @ Sigma @ K_nz.T @ rho_inv_diag @ (self.train_targets - f_mean)

            predictive_mean = self.mean_module(x) + K_xz  @ K_zz_inv @ bmu
            predictive_covar = self.covar_module(x).add_jitter() - K_xz @ (K_zz_inv - Sigma) @ K_xz.T 

            return predictive_mean, predictive_covar
    
    def ELBO_diff(self):
        
        if isinstance(self.covar_module, kernels.ScaleKernel):
            outputscale = self.covar_module.outputscale
            lengthscale = self.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.covar_module.lengthscale
            kernel_forward_fn = self.covar_module._forward_no_kernel_linop

        jitter = 1e-6
        K_nn = kernel_forward_fn(
                self.train_inputs[0]
                .div(lengthscale),
                self.train_inputs[0]
                .div(lengthscale)
            ).mul(outputscale) + jitter * torch.eye(len(self.train_targets), device=self.device)
        m_n = self.mean_module(self.train_inputs[0])

        # Partition the kernel matrix u and f into blocks
        K_nz = kernel_forward_fn(
                self.train_inputs[0]
                .div(lengthscale),
                self.inducing_points
                .div(lengthscale)
            ).mul(outputscale)
        
        K_zz = kernel_forward_fn(
            self.inducing_points.div(lengthscale),
            self.inducing_points.div(lengthscale),
        ).mul(outputscale) + jitter * torch.eye(self.inducing_points.size(0), device=self.device)

        sigma_sq = self.likelihood.noise

        chol_K_zz = torch.linalg.cholesky(K_zz)
        K_zz_inv = cholesky_solve(torch.eye(chol_K_zz.size(-1), device=self.device), chol_K_zz)

        n_data = K_nz.size(0)
        Q = K_nz @ K_zz_inv @ K_nz.T

        rho = torch.zeros(n_data, device=self.device)
        for idx in range(len(self.rho)):
            rho_i = self.rho[idx].noise
            rho[idx] = rho_i

        D = torch.diag(rho + sigma_sq)
        A = Q + D
        chol_A = torch.linalg.cholesky(A)
        A_inv = cholesky_solve(torch.eye(chol_A.size(-1), device=self.device), chol_A)
        
        expectation_term = self.train_targets - (A_inv @ (self.train_targets - m_n)).squeeze() / A_inv.diagonal().squeeze()
        var_term = 1 / A_inv.diagonal().squeeze()
        ELBO_diff = 1 / torch.sqrt(K_nn.diagonal() - Q.diagonal()) * (self.train_targets - expectation_term) - var_term
        ELBO_diff = torch.clamp(ELBO_diff, min=0)

        return ELBO_diff.squeeze()

    def get_sparse_rho(self):
        n_data = self.train_targets.size(0)
        rho_diag = torch.zeros(n_data).to(self.device)
        
        for idx in self.S:
            rho_diag[idx] = self.rho[idx].noise
        
        return rho_diag

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:

        self.eval()  
        self.likelihood.eval()
        fn = self(X)
        dist = self.likelihood(fn)

        return GPyTorchPosterior(dist)