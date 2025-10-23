import math
import torch

import sys
sys.path.append("../")

from utils.cholesky_solve import cholesky_solve

import gpytorch.kernels as kernels
import gpytorch.likelihoods as likelihoods
import gpytorch.means as means
import gpytorch.settings as settings

from src.block_diagonal_sparse_linear_operator import BlockDiagonalSparseLinearOperator
from linear_operator import operators, utils as linop_utils
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from botorch.posteriors.gpytorch import GPyTorchPosterior

class RCaGP(ExactGP):
    """Computation-aware Gaussian process."""
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        weight_function,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        projection_dim: int,
        initialization: str = "random",
    ):

        # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
        num_non_zero = train_inputs.size(-2) // projection_dim

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs[0 : num_non_zero * projection_dim],
            train_targets[0 : num_non_zero * projection_dim],
            likelihood,
        )
        
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.projection_dim = projection_dim
        self.num_non_zero = num_non_zero
        self.cholfac_gram_SKhatS = None
        
        self.weight_function = weight_function
        self.J_w = None
        self.m_w = None
        self.num_outputs=1

        non_zero_idcs = torch.arange(
            self.num_non_zero * projection_dim,
            device=train_inputs.device,
        ).reshape(self.projection_dim, -1)
        
        # Initialization of actions
        if initialization == "random":
            # Random initialization
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            # Initialize with training targets
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: self.num_non_zero * projection_dim].reshape(self.projection_dim, -1)
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(-1, 1)
            )
        elif initialization == "eigen":
            # Initialize via top eigenvectors of kernel submatrices
            with torch.no_grad():
                X = train_inputs.clone()[0 : num_non_zero * projection_dim].reshape(
                    projection_dim, num_non_zero, train_inputs.shape[-1]
                )
                K_sub_matrices = self.covar_module(X)
                _, evecs = torch.linalg.eigh(K_sub_matrices.to_dense())
            self.non_zero_action_entries = torch.nn.Parameter(evecs[:, -1])
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions_op = (
            BlockDiagonalSparseLinearOperator(  # TODO: Can we speed this up by allowing ranges as non-zero indices?
                non_zero_idcs=non_zero_idcs,
                blocks=self.non_zero_action_entries,
                size_input_dim=self.projection_dim * self.num_non_zero,
            )
        )
        
    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        if x.dim() > 2:
            batch_size, q, dim = x.shape
            predictive_mean = []
            predictive_covar = []
            for i in range(batch_size):
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
            dist = MultivariateNormal(predictive_mean, torch.absolute(predictive_covar))

        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # In training mode, just return the prior.
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            # Prior mode
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT

            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop

            sigma_sq = self.likelihood.noise
            if self.m_w is None:
                # Compute J_w and y - mw
                w = self.weight_function.W(self.train_inputs[0], self.train_targets)
                self.J_w = torch.diag(sigma_sq / 2 * 1 / w ** 2)
                m_f = self.mean_module(self.train_inputs[0])
                self.m_w = m_f + sigma_sq * self.weight_function.dylog2(self.train_inputs[0], self.train_targets)

            if self.cholfac_gram_SKhatS is None:
                # If the Cholesky factor of the gram matrix S'(K + noise)S hasn't been precomputed
                # (in the loss function), compute it.
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]),
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, 1, self.num_non_zero, self.train_inputs[0].shape[-1]),
                )

                jitter=1e-6
                device = self.train_inputs[0].device
                gram_SKS = (
                    (
                        (K_lazy @ self.actions_op.blocks.view(self.projection_dim, 1, self.num_non_zero, 1)).squeeze(-1)
                        * self.actions_op.blocks
                    )
                    .sum(-1)
                    .mul(outputscale)
                ) + torch.eye(self.num_non_zero, device=device) * jitter

                reshape_J_w = sigma_sq * self.J_w.diagonal().view(self.projection_dim, self.train_targets.size(-1) // self.projection_dim)
                StrS_diag = (self.actions_op.blocks**2 * reshape_J_w).sum(-1)  # NOTE: Assumes orthogonal actions.
                #S'(K_tilde)S
                gram_SKhatS = gram_SKS + torch.diag(StrS_diag)
                
                #L
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )
          
            #Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            #ksx
            covar_x_train_actions = (
                (
                    kernel_forward_fn(
                        x / lengthscale,
                        (self.train_inputs[0] / lengthscale).view(
                            self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]
                        ),
                    )
                    @ self.actions_op.blocks.view(self.projection_dim, self.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            #kxS @ L_inv
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            #y_bar
            actions_target = self.actions_op @ (self.train_targets - self.m_w)

            # Compressed representer weights
            #L_inv @ (S (y - mw))
            compressed_repr_weights = (
                cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            #(Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x).add_jitter() - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
            #covar = operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
            return mean, covar
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  
        self.likelihood.eval()
        dist = self.likelihood(self(X)) 

        return GPyTorchPosterior(dist)

class RCaGP_DPP(ExactGP):
    """Computation-aware Gaussian process."""
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        weight_function,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        inducing_points
    ):

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs,
            train_targets,
            likelihood,
        )
        
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.cholfac_gram_SKhatS = None
        
        self.weight_function = weight_function
        self.J_w = None
        self.m_w = None
        self.S = None
        self.num_outputs=1

        self.inducing_points = inducing_points.clone().detach()
        
    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        if x.dim() > 2:
            q, batch_size, dim = x.shape
            predictive_mean = []
            predictive_covar = []
            
            for i in range(q):
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
            dist = MultivariateNormal(predictive_mean, torch.absolute(predictive_covar))

        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # In training mode, just return the prior.
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            # Prior mode
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT
            
            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop
            sigma_sq = self.likelihood.noise

            if self.S is None:
                self.S = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale),
                    self.inducing_points
                    .div(lengthscale)
                ).mul(outputscale)
        
            if self.m_w is None:
                # Compute J_w and y - mw
                w = self.weight_function.W(self.train_inputs[0], self.train_targets)
                sigma_sq = self.likelihood.noise
                self.J_w = torch.diag(sigma_sq / 2 * 1 / w ** 2)
                m_f = self.mean_module(self.train_inputs[0])
                self.m_w = m_f + sigma_sq * self.weight_function.dylog2(self.train_inputs[0], self.train_targets)
            
            if self.cholfac_gram_SKhatS is None:
                # If the Cholesky factor of the gram matrix S'(K + noise)S hasn't been precomputed
                # (in the loss function), compute it.
                
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale),
                    self.train_inputs[0]
                    .div(lengthscale)
                ).mul(outputscale) + sigma_sq * self.J_w
                
                gram_SKhatS = self.S.T @ K_lazy @ self.S
                
                #L
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )

            #Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            #ksx
            covar_x_train_actions = kernel_forward_fn(
                    x
                    .div(lengthscale),
                    self.train_inputs[0]
                    .div(lengthscale)
            ).mul(outputscale) @ self.S

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            #kxS @ L_inv
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            #y_bar
            actions_target = self.S.T @ (self.train_targets - self.m_w)

            # Compressed representer weights
            #L_inv @ (S (y - mw))
            compressed_repr_weights = (
                cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            #(Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x).add_jitter() - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
            return mean, covar
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  
        self.likelihood.eval()
        dist = self.likelihood(self(X)) 

        return GPyTorchPosterior(dist)


class CaGP(ExactGP):
    """Computation-aware Gaussian process."""
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        projection_dim: int,
        initialization: str = "random",
    ):

        # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
        num_non_zero = train_inputs.size(-2) // projection_dim

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs[0 : num_non_zero * projection_dim],
            train_targets[0 : num_non_zero * projection_dim],
            likelihood,
        )
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.projection_dim = projection_dim
        self.num_non_zero = num_non_zero
        self.cholfac_gram_SKhatS = None
        
        self.num_outputs=1

        non_zero_idcs = torch.arange(
            self.num_non_zero * projection_dim,
            device=train_inputs.device,
        ).reshape(self.projection_dim, -1)
        # Initialization of actions
        if initialization == "random":
            # Random initialization
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            # Initialize with training targets
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: self.num_non_zero * projection_dim].reshape(self.projection_dim, -1)
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(-1, 1)
            )
        elif initialization == "eigen":
            # Initialize via top eigenvectors of kernel submatrices
            with torch.no_grad():
                X = train_inputs.clone()[0 : num_non_zero * projection_dim].reshape(
                    projection_dim, num_non_zero, train_inputs.shape[-1]
                )
                K_sub_matrices = self.covar_module(X)
                _, evecs = torch.linalg.eigh(K_sub_matrices)
            self.non_zero_action_entries = torch.nn.Parameter(evecs[:, -1])
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions_op = (
            BlockDiagonalSparseLinearOperator(  # TODO: Can we speed this up by allowing ranges as non-zero indices?
                non_zero_idcs=non_zero_idcs,
                blocks=self.non_zero_action_entries,
                size_input_dim=self.projection_dim * self.num_non_zero,
            )
        )
        
    def __call__(self, x:torch.Tensor) -> MultivariateNormal:
        if x.dim() > 2:
            batch_size, q, dim = x.shape
            predictive_mean = []
            predictive_covar = []
            
            for i in range(batch_size):
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
            dist = MultivariateNormal(predictive_mean, torch.absolute(predictive_covar))
        
        return dist
    
    def _batch_call(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # In training mode, just return the prior.
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        elif settings.prior_mode.on():
            # Prior mode
            mean = self.mean_module(x)
            covar = self.covar_module(x).add_jitter()
            return mean, covar
        
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT

            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop

            if self.cholfac_gram_SKhatS is None:
                # If the Cholesky factor of the gram matrix S'(K + noise)S hasn't been precomputed
                # (in the loss function), compute it.
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]),
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, 1, self.num_non_zero, self.train_inputs[0].shape[-1]),
                )
                
                jitter = 1e-6
                device = self.train_inputs[0].device
                gram_SKS = (
                    (
                        (K_lazy @ self.actions_op.blocks.view(self.projection_dim, 1, self.num_non_zero, 1)).squeeze(-1)
                        * self.actions_op.blocks
                    )
                    .sum(-1)
                    .mul(outputscale)
                ) + torch.eye(self.num_non_zero, device=device) * jitter
            
                StrS_diag = (self.likelihood.noise) * (self.actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
                #S'(K_tilde)S
                gram_SKhatS = gram_SKS + torch.diag(StrS_diag)
                
                #L
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )

            #Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            #ksx
            covar_x_train_actions = (
                (
                    kernel_forward_fn(
                        x / lengthscale,
                        (self.train_inputs[0] / lengthscale).view(
                            self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]
                        ),
                    )
                    @ self.actions_op.blocks.view(self.projection_dim, self.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            #kxS @ L_inv
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            actions_target = self.actions_op @ (self.train_targets - self.mean_module(self.train_inputs[0]))

            # Compressed representer weights
            #L_inv @ (S (y - mw))
            compressed_repr_weights = (
                cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            #(Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x).add_jitter() - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
            return mean, covar
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  
        self.likelihood.eval()
        dist = self.likelihood(self(X)) 

        return GPyTorchPosterior(dist)