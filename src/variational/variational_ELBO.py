import torch
import math
import gpytorch.kernels as kernels
from linear_operator import operators, utils as linop_utils

import sys 
sys.path.append("../")

from linear_operator import operators, utils as linop_utils
from utils.cholesky_solve import cholesky_solve
from gpytorch.mlls import MarginalLogLikelihood

class RCSVGPVariationalELBO(MarginalLogLikelihood):
    
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
        
    def forward(
            self,
            q,
            y_train,
            x_train):
        
        inducing_points = self.model.inducing_points
        num_inducing = inducing_points.size(-2)
        full_covar = q.covariance_matrix
        full_mean = q.mean

        jitter=1e-6

        # Partition the kernel matrix u and f into blocks
        m_f = full_mean[num_inducing:]
        K_fu = full_covar[num_inducing:, :num_inducing]
        K_uu = full_covar[:num_inducing, :num_inducing]
        K_ff = full_covar[num_inducing:, num_inducing:]
        K_ff_diag = K_ff.diagonal()

        # Compute term 1
        sigma_sq = self.likelihood.noise
        chol_factor = torch.linalg.cholesky(K_uu)
        K_uu_inv = cholesky_solve(torch.eye(chol_factor.size(-1)).to(chol_factor.device), chol_factor)
        w = self.model.weight_function.W(self.model.train_inputs[0], self.model.train_targets)
        J_w_diag = sigma_sq / 2 * 1 / (w ** 2 + jitter)
        J_w_inv_diag = 1.0 / (J_w_diag + jitter)
        J_w_inv_sqrt_diag = torch.sqrt(J_w_inv_diag)
        term1 = 0.5 * torch.sum(1 / (sigma_sq + jitter) * J_w_inv_sqrt_diag * (K_ff_diag - torch.einsum("nm,mk,nk->n", K_fu, K_uu_inv, K_fu)) * J_w_inv_sqrt_diag)

        # Compute C term
        nabla_y = torch.sum(w ** 2) + torch.sum(self.model.train_targets * 2 * w * self.model.weight_function.dy(self.model.train_inputs[0], self.model.train_targets))
        C = 1/(sigma_sq + jitter) * torch.sum(self.model.train_targets * (2 * 1 / (sigma_sq + jitter) *  w ** 2 ) * self.model.train_targets) - 4 * sigma_sq * nabla_y
        C = 0.5 * C

        # Compute term 3
        m_w = m_f + sigma_sq * self.model.weight_function.dylog2(self.model.train_inputs[0], self.model.train_targets)
        nu = 1 / (sigma_sq + jitter)  * J_w_inv_diag * (self.model.train_targets - m_w)
        #temp = K_uu + (1 / sigma_sq + jitter) * K_fu.T @ J_w_inv @ K_fu
        outer = torch.mm(K_fu.T, J_w_inv_diag.reshape(-1, 1) * K_fu)
        temp = K_uu + 1 / (sigma_sq + jitter) * outer

        temp_factor = torch.linalg.cholesky(temp)
        temp_inv = cholesky_solve(torch.eye(temp_factor.size(-1)).to(temp_factor.device), temp_factor)
        term3 = 0.5 * torch.sum(torch.sum(nu.reshape(-1, 1) * (K_fu @ temp_inv @ K_fu.T), dim=0) * nu)

        # Compute term 4
        log_det_K_uu = 4 * torch.sum(torch.log(chol_factor.diagonal()))
        log_det_temp = 2 * torch.sum(torch.log(temp_factor.diagonal()))
        term4 = 0.5 * (log_det_K_uu - log_det_temp)

        n_data = len(self.model.train_targets)
        elbo = (- term1 - C + term3 + term4) / n_data

        return elbo.squeeze()

class RCaGPVariationalELBO(MarginalLogLikelihood):
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
    
    def forward(self,
            q,
            y_train,
            x_train,
            ):

        n_data = self.model.train_targets.shape[0]
        m_f = q.mean[:n_data]
        var_f = q.variance[:n_data]
        sigma_sq = self.likelihood.noise
        jitter=1e-6
    
        del self.model.J_w
        del self.model.m_w

        w = self.model.weight_function.W(self.model.train_inputs[0], self.model.train_targets)
        w =  torch.nan_to_num(w, nan=1.0)
        J_w_diag = sigma_sq / 2 * 1 / w ** 2 + jitter
        J_w_inv_diag = 1.0 / J_w_diag + jitter
        J_w_inv_sqrt_diag = torch.sqrt(J_w_inv_diag)
        m_w = m_f + (sigma_sq + jitter) * self.model.weight_function.dylog2(self.model.train_inputs[0], self.model.train_targets)

        self.model.J_w = torch.diag(J_w_diag)
        self.model.m_w = m_w.clone().detach()

        # Kernel forward and hyperparameters
        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.model.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.model.covar_module.lengthscale
            kernel_forward_fn = self.model.covar_module._forward_no_kernel_linop

        del self.model.cholfac_gram_SKhatS

        K_lazy = kernel_forward_fn(
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.projection_dim, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.projection_dim, 1, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
        )

        StK_block = (K_lazy @ self.model.actions_op.blocks.view(self.model.projection_dim, 1, self.model.num_non_zero, 1)).squeeze(-1)

        gram_SKS = (
            (
                StK_block
                * (self.model.actions_op.blocks + jitter)
            )
            .sum(-1)
            .mul(outputscale)
        )

        reshape_J_w = sigma_sq * J_w_diag.view(self.model.projection_dim, self.model.train_targets.size(-1) // self.model.projection_dim)
        StrS_diag = (self.model.actions_op.blocks**2 * reshape_J_w + jitter).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(StrS_diag)

        #(SK_hatS)^{-1}
        cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
            gram_SKhatS.to(dtype=torch.float64), upper=False
        )
        cholfac_gram_SKhatS = cholfac_gram_SKhatS + torch.eye(cholfac_gram_SKhatS.size(-1)).to(cholfac_gram_SKhatS.device) * jitter

        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        #ksx
        covar_x_train_actions = (
            StK_block.view(self.model.projection_dim, self.model.projection_dim * self.model.num_non_zero)
            .mul(outputscale)
            .mT
        )

        #ksx @ L_inv
        covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
        ).mT

        #y_bar
        #actions_target = self.model.actions_op @ (self.model.train_targets - m_w)
        actions_target = self.model.actions_op._matmul(
            torch.atleast_2d(self.model.train_targets - m_w).mT
        ).squeeze(-1)

        #L_inv @ y_bar
        compressed_repr_weights = (
            cholesky_solve(
                actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
            )
            .squeeze(-1)
            .to(self.model.train_inputs[0].dtype)
        )

        q_mean = m_f + covar_x_train_actions @ torch.atleast_1d(compressed_repr_weights)
        #q_variance = K - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
        q_var_diag = var_f - (covar_x_train_actions_cholfac_inv ** 2).sum(dim=-1)

        #Compute the loss term
        term1 = 0.5 * ( (1 / sigma_sq + jitter) * J_w_inv_sqrt_diag * q_var_diag * J_w_inv_sqrt_diag).sum()

        term2 = 0.5 * torch.sum( (1 / sigma_sq + jitter) * q_mean * J_w_inv_diag * q_mean )

        nu = (1 / sigma_sq + jitter)  * J_w_inv_diag * (self.model.train_targets - m_w)
        term3 = torch.sum(q_mean * nu)

        nabla = torch.sum(w ** 2) + torch.sum(self.model.train_targets * 2 * w * self.model.weight_function.dy(self.model.train_inputs[0], self.model.train_targets))
        C =  (1/sigma_sq + jitter) * torch.sum(self.model.train_targets * (2 * (1 / sigma_sq + jitter) *  w ** 2 ) * self.model.train_targets) - 4 * sigma_sq * (nabla)   
        C  = 0.5 * C

        loss_term = - term1 - term2 + term3 - C

        #KL-term
        #S'(K_tilde)S
        logdet_SKhatS = 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))

        vSKSv = torch.inner(compressed_repr_weights, (gram_SKS @ compressed_repr_weights))

        SKhatS_inv_SKS = cholesky_solve(gram_SKS, cholfac_gram_SKhatS, upper=False)
        tr_SKhatS_inv_SKS = torch.sum(SKhatS_inv_SKS.diagonal())

        logdet_StrS = torch.sum( torch.log(StrS_diag) )

        kl_term = vSKSv + logdet_SKhatS - logdet_StrS - tr_SKhatS_inv_SKS

        n_data = len(self.model.train_targets)
        elbo = (loss_term - 0.5 * kl_term) / n_data
        return elbo.squeeze()

class CaGPVariationalELBO(MarginalLogLikelihood):
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
    
    def forward(self,
            q,
            y_train,
            x_train,
            ):

        n_data = self.model.train_targets.shape[0]
        m_f = q.mean[:n_data]
        var_f = q.variance[:n_data]
        sigma_sq = self.likelihood.noise
        jitter=1e-6

        # Kernel forward and hyperparameters
        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.model.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.model.covar_module.lengthscale
            kernel_forward_fn = self.model.covar_module._forward_no_kernel_linop

        del self.model.cholfac_gram_SKhatS

        K_lazy = kernel_forward_fn(
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.projection_dim, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
            self.model.train_inputs[0]
            .div(lengthscale)
            .view(self.model.projection_dim, 1, self.model.num_non_zero, self.model.train_inputs[0].shape[-1]),
        )
        
        StK_block = (K_lazy @ self.model.actions_op.blocks.view(self.model.projection_dim, 1, self.model.num_non_zero, 1)).squeeze(-1)
        gram_SKS = (
            (
                StK_block
                * self.model.actions_op.blocks + jitter
            )
            .sum(-1)
            .mul(outputscale)
        )

        StrS_diag = sigma_sq * (self.model.actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(StrS_diag)

        #(SK_hatS)^{-1}
        cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
            gram_SKhatS.to(dtype=torch.float64), upper=False
        )
        cholfac_gram_SKhatS = cholfac_gram_SKhatS + torch.eye(cholfac_gram_SKhatS.size(-1)).to(cholfac_gram_SKhatS.device) * jitter

        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        #ksx
        covar_x_train_actions = (
            StK_block.view(self.model.projection_dim, self.model.projection_dim * self.model.num_non_zero)
            .mul(outputscale)
            .mT
        )

        #ksx @ L_inv
        covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
        ).mT

        #y_bar
        #actions_target = self.model.actions_op @ (self.model.train_targets - m_w)
        actions_target = self.model.actions_op._matmul(
            torch.atleast_2d(self.model.train_targets - m_f).mT
        ).squeeze(-1)

        #L_inv @ y_bar
        compressed_repr_weights = (
            cholesky_solve(
                actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
            )
            .squeeze(-1)
            .to(self.model.train_inputs[0].dtype)
        )

        q_mean = m_f + covar_x_train_actions @ torch.atleast_1d(compressed_repr_weights)
        #q_variance = K - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)
        q_var_diag = var_f - (covar_x_train_actions_cholfac_inv ** 2).sum(dim=-1)
        n_data = len(self.model.train_targets)

        #Compute the loss term
        loss_term = -0.5 * (
            n_data * torch.log(self.likelihood.noise)
            + 1
            / self.likelihood.noise
            * (torch.linalg.vector_norm(self.model.train_targets - q_mean) ** 2 + torch.sum(q_var_diag) )
            + n_data * torch.log(torch.as_tensor(2 * math.pi))
        )

        #KL-term
        #S'(K_tilde)S
        logdet_SKhatS = 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))

        vSKSv = torch.inner(compressed_repr_weights, (gram_SKS @ compressed_repr_weights))

        SKhatS_inv_SKS = cholesky_solve(gram_SKS, cholfac_gram_SKhatS, upper=False)
        tr_SKhatS_inv_SKS = torch.sum(SKhatS_inv_SKS.diagonal())

        logdet_StrS = torch.sum( torch.log(StrS_diag) )

        kl_term = vSKSv + logdet_SKhatS - logdet_StrS - tr_SKhatS_inv_SKS

        
        elbo = (loss_term - 0.5 * kl_term) / n_data
        return elbo.squeeze()