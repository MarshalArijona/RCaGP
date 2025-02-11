import torch
import gpytorch.settings as settings
import gpytorch.kernels as kernels

import sys 
sys.path.append("../")

from utils.cholesky_solve import cholesky_solve
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior

class RCSVGP(ExactGP):
    def __init__(self, 
            inducing_points,
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

        self.inducing_points = torch.nn.Parameter(inducing_points.clone())
        self.num_inducing = self.inducing_points.size(-2)
        self.weight_function = weight_function
        self.f_inputs = torch.cat([self.inducing_points, self.train_inputs[0]], dim=-2)
        self.num_outputs = 1

    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        if x.dim() > 2:
            jitter=1e-8
            shape = x.shape
            x_reshape = x.reshape(shape[0] * shape[1], shape[2])
            raw_predictive_mean, raw_predictive_covar = self._batch_call(x_reshape)
            raw_predictive_covar = raw_predictive_covar.to_dense() + jitter * torch.eye(raw_predictive_covar.shape[-1], device=raw_predictive_covar.device)
            predictive_mean = raw_predictive_mean.reshape(shape[0], shape[1])
            #predictive_covar = torch.stack([raw_predictive_covar[i*shape[1]:(i+1)*shape[1], i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])], dim=0)
            predictive_covar = raw_predictive_covar.diagonal().reshape(shape[0], shape[1], shape[1])
        else:
            predictive_mean, predictive_covar =  self._batch_call(x)

        return MultivariateNormal(predictive_mean, predictive_covar)
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        jitter=1e-6
        if self.training:
            inducing_points = self.inducing_points
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            mean = self.mean_module(full_inputs)
            covar = self.covar_module(full_inputs).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            inducing_points = self.inducing_points
            full_inputs = torch.cat([inducing_points, x], dim=-2)
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
            
            f_mean = self.mean_module(self.f_inputs)
            #f_covar = self.covar_module(self.f_inputs)
            f_covar = kernel_forward_fn(
                    self.f_inputs.div(lengthscale),
                    self.f_inputs.div(lengthscale),
                )

            # Partition the kernel matrix u and f into blocks
            m_f = f_mean[self.num_inducing:]
            K_fu = f_covar[self.num_inducing:, :self.num_inducing]

            # Compute J_w and y - mw
            w = self.weight_function.W(self.train_inputs[0], self.train_targets)
            sigma_sq = self.likelihood.noise
            J_w = torch.diag(sigma_sq / 2  * 1 / w ** 2 + jitter )
            m_w = m_f + self.weight_function.dylog2(self.train_inputs[0], self.train_targets)
            y_min_mw = self.train_targets - m_w

            # Compute the kernel matrix between data test and inducing points
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            #full_covar = self.covar_module(full_inputs)

            full_covar = kernel_forward_fn(
                    full_inputs.div(lengthscale),
                    full_inputs.div(lengthscale),
                )

            # Partition the kernel matrix u and s into blocks
            K_uu = full_covar[:self.num_inducing, :self.num_inducing] + torch.eye(self.num_inducing, device=full_covar.device) * jitter
            K_su = full_covar[self.num_inducing:, :self.num_inducing]
            K_ss = full_covar[self.num_inducing:, self.num_inducing:] + torch.eye(x.shape[-2], device=full_covar.device) * jitter

            # Compute predictive posterior
            predictive_mean, predictive_covar = self._approximate_posterior(
                x,
                K_uu, 
                K_ss,
                K_su,
                J_w,
                K_fu,
                sigma_sq,
                m_f,
                y_min_mw
            )
            return predictive_mean, predictive_covar
        
    def _approximate_posterior(
        self,
        x,
        K_uu, 
        K_ss,
        K_su,
        J_w,
        K_fu,
        sigma_sq,
        m_f,
        y_min_mw
    ):

        """
        Helper function to compute the approximate posterior.
        """

        # Solve for the inverse of the inducing covariance
        chol_factor = torch.linalg.cholesky(K_uu)
        K_uu_inv = cholesky_solve(torch.eye(chol_factor.size(-1), device=chol_factor.device), chol_factor)
        jitter=1e-6

        #Compute variational distribution mu_u and Sigma_u
        J_w_inv = torch.diag(1.0 / J_w.diagonal())
        P_u = K_uu + K_fu.T @ ((1 / sigma_sq + jitter) * J_w_inv) @ K_fu
        P_u_factor = torch.linalg.cholesky(P_u)
        P_u_inv = cholesky_solve(torch.eye(P_u_factor.size(-1), device=P_u_factor.device), P_u_factor)
        mu_u = K_uu @ P_u_inv @ K_fu.T @ ((1 / sigma_sq + jitter) * J_w_inv) @ y_min_mw
        Sigma_u = K_uu @ P_u_inv @ K_uu

        #Compute mean posterior
        phi_u = K_uu_inv @ K_su.T
        predictive_mean = self.mean_module(x) + phi_u.T @ mu_u

        # Compute the covariance of the posterior
        predictive_covar = K_ss - phi_u.T @ (K_uu - Sigma_u) @ phi_u

        return predictive_mean, predictive_covar
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:

        self.eval()  
        self.likelihood.eval()
        fn = self(X)
        dist = self.likelihood(fn)

        return GPyTorchPosterior(dist)