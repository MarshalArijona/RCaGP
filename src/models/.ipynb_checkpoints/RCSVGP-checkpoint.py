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
            weight_function,
            learn_inducing_locations=True):
        
        super().__init__(
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=likelihood,
        )

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.learn_inducing_locations = learn_inducing_locations
        if self.learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points.clone())
        else:
            self.inducing_points = inducing_points.clone()
        self.num_inducing = self.inducing_points.size(-2)
        self.weight_function = weight_function
        self.f_inputs = torch.cat([self.inducing_points, self.train_inputs[0]], dim=-2)
        self.num_outputs = 1
        self.device = train_x.device

    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        
        '''
        - If dim is greater than 2, reshape the inputs. Run the forward method of RCSVGP.
        - For now, it only works for the case (batch, 1, dim) --> single-batch acquisition function.
        '''

        if x.dim() > 2:
            batch, _, _ = x.shape

            predictive_mean = []
            predictive_covar = []
            for i in range(batch):
                batch_x = x[i]
                q_predictive_mean, q_predictive_covar = self._batch_call(batch_x)
                
                predictive_mean.append(q_predictive_mean)
                predictive_covar.append(q_predictive_covar.to_dense())
            
            predictive_mean = torch.stack(predictive_mean)
            predictive_covar = torch.stack(predictive_covar)
        else:
            predictive_mean, predictive_covar =  self._batch_call(x)

        try:
            dist = MultivariateNormal(predictive_mean, predictive_covar)
        except:
            
            new_cov = torch.absolute(predictive_covar)
            dist = MultivariateNormal(predictive_mean, new_cov)

        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
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

            jitter = 1e-4            
            f_mean = self.mean_module(self.f_inputs)
            n_data = self.f_inputs.shape[0]
            f_covar = kernel_forward_fn(
                    self.f_inputs.div(lengthscale),
                    self.f_inputs.div(lengthscale),
                ).mul(outputscale) + torch.eye(n_data, device=self.device) * jitter

            # Partition the kernel matrix u and f into blocks
            m_f = f_mean[self.num_inducing:]
            K_fu = f_covar[self.num_inducing:, :self.num_inducing]

            #Compute J_w and y - mw
            w = self.weight_function.W(self.train_inputs[0], self.train_targets)
            sigma_sq = self.likelihood.noise
            J_w = torch.diag(sigma_sq / 2  * 1 / w ** 2)
            
            m_w = m_f + self.weight_function.dylog2(self.train_inputs[0], self.train_targets)
            y_min_mw = self.train_targets - m_w

            # Compute the kernel matrix between data test and inducing points
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            n_pred = full_inputs.shape[0]
            full_covar = kernel_forward_fn(
                    full_inputs.div(lengthscale),
                    full_inputs.div(lengthscale),
                ).mul(outputscale) + torch.eye(n_pred, device=self.device) * jitter

            # Partition the kernel matrix u and s into blocks
            K_uu = full_covar[:self.num_inducing, :self.num_inducing]
            K_su = full_covar[self.num_inducing:, :self.num_inducing]
            K_ss = full_covar[self.num_inducing:, self.num_inducing:]

            # Compute predictive posterior
            predictive_mean, predictive_covar = self._approximate_posterior(
                x,
                K_uu, 
                K_ss,
                K_su,
                J_w,
                K_fu,
                sigma_sq,
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
        y_min_mw
    ):

        """
        Helper function to compute the approximate posterior.
        """

        #Solve for the inverse of the inducing points gram matrix
        chol_factor = torch.linalg.cholesky(K_uu)
        K_uu_inv = cholesky_solve(torch.eye(chol_factor.size(-1), device=self.device), chol_factor)

        #Calculate the variational distribution parameters mu_u and Sigma_u
        jitter = 1e-4
        J_w_inv = torch.diag(1.0 / J_w.diagonal())
        P_u = K_uu + 1 / sigma_sq * K_fu.T @ J_w_inv @ K_fu
        P_u += torch.eye(P_u.shape[-1], device=self.device) * jitter

        try:
            P_u_factor = torch.linalg.cholesky(P_u)
        except:
            P_u_factor = torch.linalg.cholesky(P_u + torch.eye(P_u.shape[-1], device=self.device) * 1e13)
            
        P_u_inv = cholesky_solve(torch.eye(P_u_factor.size(-1), device=self.device), P_u_factor)
        mu_u = 1 / sigma_sq * K_uu @ P_u_inv @ K_fu.T @ J_w_inv @ y_min_mw
        Sigma_u = K_uu @ P_u_inv @ K_uu

        self.mu_u = mu_u.clone().detach()
        self.Sigma_u = Sigma_u.clone().detach()

        #Compute the predictive mean
        phi_u = K_uu_inv @ K_su.T
        predictive_mean = self.mean_module(x) + phi_u.T @ mu_u

        #Compute the predictive covariance
        predictive_covar = self.covar_module(x).add_jitter() - phi_u.T @ (K_uu - Sigma_u) @ phi_u

        return predictive_mean, predictive_covar
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:

        self.eval()  
        self.likelihood.eval()
        fn = self(X)
        dist = self.likelihood(fn)

        return GPyTorchPosterior(dist)