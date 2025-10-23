import torch
import math
import gpytorch.kernels as kernels
from linear_operator import operators, utils as linop_utils

import sys 
sys.path.append("../")

from linear_operator import operators, utils as linop_utils
from utils.cholesky_solve import cholesky_solve
from gpytorch.mlls import MarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError

class RCGPVariationalELBO(MarginalLogLikelihood):
    
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
        
    def forward(
            self,
            p,
            y):

        m_f = p.mean.flatten()
        K_f = p.covariance_matrix

        # Compute J_w and y - mw
        w = self.model.weight_function.W(self.model.train_inputs[0], self.model.train_targets)
        sigma_sq = self.model.likelihood.noise
        J_w = torch.diag(sigma_sq / 2  * 1 / w ** 2)
        
        m_w = m_f + self.model.weight_function.dylog2(self.model.train_inputs[0], self.model.train_targets)
        z = y - m_w

        #Solve for the inverse of the gram matrix
        chol_factor = torch.linalg.cholesky(K_f + sigma_sq * J_w)
        K_inv = cholesky_solve(torch.eye(chol_factor.size(-1), device=self.model.train_inputs[0].device), chol_factor)
        temp = K_inv @ z

        predictive_mu = z + m_f - temp * 1 / K_inv.diagonal()
        predictive_sigma = 1 / K_inv.diagonal() - sigma_sq ** 2 / 2 * 1 / w ** 2
        
        sum = -0.5 * torch.log(2 * torch.pi * predictive_sigma) - 0.5 * (y - predictive_mu)**2 / predictive_sigma
        loo = torch.mean(sum).squeeze()
    
        return loo

class RCSVGPVariationalELBO(MarginalLogLikelihood):
    
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
        
    def forward(
            self,
            q):
        
        inducing_points = self.model.inducing_points
        num_inducing = inducing_points.size(-2)
        
        full_mean = q.mean.flatten()
        full_covar = q.covariance_matrix

        train_inputs = self.model.train_inputs[0]
        train_targets = self.model.train_targets.flatten()

        # Partition the kernel matrix u and f into blocks
        m_f = full_mean[num_inducing:]
        K_fu = full_covar[num_inducing:, :num_inducing]
        K_uu = full_covar[:num_inducing, :num_inducing]
        K_uu = torch.diag(K_uu.diagonal())
        K_ff = full_covar[num_inducing:, num_inducing:]
        K_ff_diag = K_ff.diagonal()

        # Compute term 1
        sigma_sq = self.likelihood.noise
        
        Kuu_factor = torch.linalg.cholesky(K_uu)
        K_uu_inv = cholesky_solve(torch.eye(Kuu_factor.size(-1)).to(Kuu_factor.device), Kuu_factor)
        
        w = self.model.weight_function.W(train_inputs, train_targets)
        J_w_diag = sigma_sq / 2 * 1 / w ** 2
        J_w_inv_diag = 1.0 / J_w_diag
        J_w_inv_sqrt_diag = torch.sqrt(J_w_inv_diag)
        term1 = 0.5 * torch.sum(1 / sigma_sq * J_w_inv_sqrt_diag * (K_ff_diag - torch.einsum("nm,mk,nk->n", K_fu, K_uu_inv, K_fu)) * J_w_inv_sqrt_diag)

        # Compute C term
        nabla_y = torch.sum(w ** 2) + torch.sum(train_targets * 2 * w * self.model.weight_function.dy(train_inputs, train_targets))
        C = 1 / sigma_sq * torch.sum(train_targets * (2 * 1 / sigma_sq *  w ** 2 ) * train_targets) - 4 * sigma_sq * nabla_y
        C *= 0.5

        # Compute term 3
        m_w = m_f + sigma_sq * self.model.weight_function.dylog2(train_inputs, train_targets)
        nu = 1 / sigma_sq  * J_w_inv_diag * (train_targets - m_w + m_f)
        outer = torch.mm(K_fu.T, J_w_inv_diag.reshape(-1, 1) * K_fu)
        temp = K_uu + 1 / sigma_sq * outer
        temp_factor = torch.linalg.cholesky(temp)
        temp_inv = cholesky_solve(torch.eye(temp_factor.size(-1)).to(temp_factor.device), temp_factor)
        term3 = 0.5 * torch.sum(torch.sum(nu.reshape(-1, 1) * (K_fu @ temp_inv @ K_fu.T), dim=0) * nu)

        # Compute term 4
        log_det_K_uu = 4 * torch.sum(torch.log(Kuu_factor.diagonal()))
        log_det_temp = 2 * torch.sum(torch.log(temp_factor.diagonal()))
        term4 = 0.5 * (log_det_K_uu - log_det_temp)

        elbo = - term1 - C + term3 + term4
        return elbo.squeeze()

class RCaGPVariationalELBO(MarginalLogLikelihood):
    def __init__(self, 
            likelihood, 
            model,
            alpha=1.0):
        
        super().__init__(likelihood, model)
        self.alpha = alpha
    
    def forward(self,
            q
            ):

        train_targets = self.model.train_targets.flatten()
        train_inputs = self.model.train_inputs[0]
        n_data = train_targets.shape[0]
        m_f = q.mean[:n_data].flatten()
        var_f = q.variance[:n_data].flatten()
        sigma_sq = self.likelihood.noise
    
        del self.model.J_w
        del self.model.m_w
        
        w = self.model.weight_function.W(train_inputs, train_targets)
        J_w_diag = sigma_sq / 2 * 1 / w ** 2
        J_w_inv_diag = 1.0 / J_w_diag
        J_w_inv_sqrt_diag = torch.sqrt(J_w_inv_diag)
        m_w = m_f + sigma_sq * self.model.weight_function.dylog2(train_inputs, train_targets)

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
            train_inputs
            .div(lengthscale)
            .reshape(self.model.projection_dim, self.model.num_non_zero, train_inputs.shape[-1]),
            train_inputs
            .div(lengthscale)
            .reshape(self.model.projection_dim, 1, self.model.num_non_zero, train_inputs.shape[-1]),
        )

        StK_block = (K_lazy @ self.model.actions_op.blocks.view(self.model.projection_dim, 1, self.model.num_non_zero, 1)).squeeze(-1)

        gram_SKS = (
            (
                StK_block
                * (self.model.actions_op.blocks)
            )
            .sum(-1)
            .mul(outputscale)
        )

        reshape_J_w = sigma_sq * J_w_diag.reshape(self.model.projection_dim, train_targets.shape[0] // self.model.projection_dim)
        StrS_diag = (self.model.actions_op.blocks**2 * reshape_J_w).sum(-1)  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(StrS_diag)

        #(SK_hatS)^{-1}
        cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
            gram_SKhatS.to(dtype=torch.float64), upper=False
        )
        cholfac_gram_SKhatS = cholfac_gram_SKhatS

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
        actions_target = self.model.actions_op._matmul(
            torch.atleast_2d(train_targets - m_w).mT
        ).squeeze(-1)

        #L_inv @ y_bar
        compressed_repr_weights = (
            cholesky_solve(
                actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
            )
            .squeeze(-1)
            .to(train_inputs.dtype)
        )

        q_mean = m_f + covar_x_train_actions @ torch.atleast_1d(compressed_repr_weights)
        q_var_diag = var_f - (covar_x_train_actions_cholfac_inv ** 2).sum(dim=-1)

        #Compute the loss term
        term1 = 0.5 * torch.sum( 1 / sigma_sq * J_w_inv_sqrt_diag * q_var_diag * J_w_inv_sqrt_diag)

        term2 = 0.5 * torch.sum( 1 / sigma_sq * q_mean * J_w_inv_diag * q_mean)

        nu = 1 / sigma_sq  * J_w_inv_diag * (train_targets - m_w + m_f)
        term3 = torch.sum(q_mean * nu)

        nabla = torch.sum(w ** 2) + torch.sum(train_targets * 2 * w * self.model.weight_function.dy(train_inputs, train_targets))
        C =  1/sigma_sq * torch.sum(train_targets * (2 * 1 / sigma_sq *  w ** 2 ) * train_targets) - 4 * sigma_sq * (nabla)   
        C  = 0.5 * C

        loss_term = - term1 - term2 + term3 - C

        #KL-term
        #S'(K_tilde)S
        logdet_SKhatS = 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))

        vSKSv = torch.inner(compressed_repr_weights, (gram_SKS @ compressed_repr_weights))

        SKhatS_inv_SKS = cholesky_solve(gram_SKS, cholfac_gram_SKhatS, upper=False)
        tr_SKhatS_inv_SKS = torch.sum(SKhatS_inv_SKS.diagonal())

        logdet_StrS = torch.sum( torch.log(StrS_diag) )

        kl_term = 0.5 * (vSKSv + logdet_SKhatS - logdet_StrS - tr_SKhatS_inv_SKS)

        elbo = loss_term - self.alpha * kl_term
        return elbo.squeeze()

class RCaGP_DPPVariationalELBO(MarginalLogLikelihood):
    def __init__(self, 
            likelihood, 
            model,
            alpha=1.0):
        
        super().__init__(likelihood, model)
        self.alpha = alpha
    
    def forward(self,
            q
            ):

        train_targets = self.model.train_targets.flatten()
        train_inputs = self.model.train_inputs[0]
        n_data = train_targets.shape[0]
        m_f = q.mean[:n_data].flatten()
        var_f = q.variance[:n_data].flatten()
        sigma_sq = self.likelihood.noise

        del self.model.J_w
        del self.model.m_w
        
        w = self.model.weight_function.W(train_inputs, train_targets)
        J_w_diag = sigma_sq / 2 * 1 / w ** 2
        J_w_inv_diag = 1.0 / J_w_diag
        J_w_inv_sqrt_diag = torch.sqrt(J_w_inv_diag)
        m_w = m_f + sigma_sq * self.model.weight_function.dylog2(train_inputs, train_targets)
        J_w = torch.diag(J_w_diag)

        self.model.J_w = J_w.clone().detach()
        self.model.m_w = m_w.clone().detach()

        if isinstance(self.model.covar_module, kernels.ScaleKernel):
            outputscale = self.model.covar_module.outputscale
            lengthscale = self.model.covar_module.base_kernel.lengthscale
            kernel_forward_fn = self.model.covar_module.base_kernel._forward_no_kernel_linop
        else:
            outputscale = 1.0
            lengthscale = self.model.covar_module.lengthscale
            kernel_forward_fn = self.model.covar_module._forward_no_kernel_linop

        del self.model.S
        S = kernel_forward_fn(
                    train_inputs
                    .div(lengthscale),
                    self.model.inducing_points
                    .div(lengthscale)
                ).mul(outputscale)
        self.model.S = S.clone().detach()

        del self.model.cholfac_gram_SKhatS

        K_lazy = kernel_forward_fn(
                    train_inputs
                    .div(lengthscale),
                    train_inputs
                    .div(lengthscale)
                ).mul(outputscale) + sigma_sq * J_w

        gram_KhatS = K_lazy @ S
        gram_SKhatS = S.T @ gram_KhatS

        #(SK_hatS)^{-1}
        try:
            cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                gram_SKhatS.to(dtype=torch.float64), upper=False
            )
        except NotPSDError:
            jitter = 1e7 * torch.eye(gram_SKhatS.size(0), device=gram_SKhatS.device)
            cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    (gram_SKhatS + jitter).to(dtype=torch.float64), upper=False
                )
  
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        #ksx
        covar_x_train_actions = gram_KhatS

        #ksx @ L_inv
        covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
        ).mT

        #y_bar
        actions_target = S.T @ (train_targets - m_w)

        #L_inv @ y_bar
        compressed_repr_weights = (
            cholesky_solve(
                actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
            )
            .squeeze(-1)
            .to(train_inputs.dtype)
        )

        q_mean = m_f + covar_x_train_actions @ torch.atleast_1d(compressed_repr_weights)
        q_var_diag = var_f - (covar_x_train_actions_cholfac_inv ** 2).sum(dim=-1)

        #Compute the loss term
        term1 = 0.5 * torch.sum( 1 / sigma_sq * J_w_inv_sqrt_diag * q_var_diag * J_w_inv_sqrt_diag)

        term2 = 0.5 * torch.sum( 1 / sigma_sq * q_mean * J_w_inv_diag * q_mean)

        nu = 1 / sigma_sq  * J_w_inv_diag * (train_targets - m_w + m_f)
        term3 = torch.sum(q_mean * nu)

        nabla = torch.sum(w ** 2) + torch.sum(train_targets * 2 * w * self.model.weight_function.dy(train_inputs, train_targets))
        C =  1/sigma_sq * torch.sum(train_targets * (2 * 1 / sigma_sq *  w ** 2 ) * train_targets) - 4 * sigma_sq * (nabla)   
        C  = 0.5 * C

        loss_term = - term1 - term2 + term3 - C

        #KL-term
        #S'(K_tilde)S
        logdet_SKhatS = 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))

        vSKSv = torch.inner(compressed_repr_weights, (gram_SKhatS @ compressed_repr_weights))

        SKhatS_inv_SKS = cholesky_solve(gram_SKhatS, cholfac_gram_SKhatS, upper=False)
        tr_SKhatS_inv_SKS = torch.sum(SKhatS_inv_SKS.diagonal())

        StrS_diag = (S ** 2).sum(dim=-2)
        logdet_StrS = torch.sum(torch.log(StrS_diag))

        kl_term = 0.5 * (vSKSv + logdet_SKhatS - logdet_StrS - tr_SKhatS_inv_SKS)

        elbo = loss_term - self.alpha * kl_term
        return elbo.squeeze()

class CaGPVariationalELBO(MarginalLogLikelihood):
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)
    
    def forward(self,
            q,
            ):

        n_data = self.model.train_targets.shape[0]
        m_f = q.mean[:n_data]
        var_f = q.variance[:n_data]
        sigma_sq = self.likelihood.noise

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
                * self.model.actions_op.blocks
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
        cholfac_gram_SKhatS = cholfac_gram_SKhatS + torch.eye(cholfac_gram_SKhatS.size(-1)).to(cholfac_gram_SKhatS.device)

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
        q_var_diag = var_f - (covar_x_train_actions_cholfac_inv ** 2).sum(dim=-1)

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

        elbo = loss_term - 0.5 * kl_term
        return elbo.squeeze()

class RRPVariationalELBO(MarginalLogLikelihood):
    
    def __init__(self, 
            likelihood, 
            model):
        
        super().__init__(likelihood, model)

    def forward(
            self,
            q):
              
        num_inducing = self.model.inducing_points.size(-2)
        full_mean = q.mean.flatten()
        full_covar = q.covariance_matrix

        # Partition the kernel matrix u and f into blocks
        m_n = full_mean[num_inducing:]
        K_nz = full_covar[num_inducing:, :num_inducing]
        K_zz = full_covar[:num_inducing, :num_inducing]
        K_nn = full_covar[num_inducing:, num_inducing:]

        sigma_sq = self.likelihood.noise

        chol_K_zz = torch.linalg.cholesky(K_zz)
        K_zz_inv = cholesky_solve(torch.eye(chol_K_zz.size(-1), device=self.model.device), chol_K_zz)

        Q = K_nz @ K_zz_inv @ K_nz.T
        rho_diag = self.model.get_sparse_rho() + sigma_sq
        D = torch.diag(rho_diag)
        A = Q + D
        chol_A = torch.linalg.cholesky(A)
        log_det_A = 2 * torch.log(chol_A.diagonal()).sum()
        A_inv = cholesky_solve(torch.eye(chol_A.size(-1), device=self.model.device), chol_A)

        quadratic_term = (self.model.train_targets - m_n).reshape(-1, 1).T @ A_inv @ (self.model.train_targets - m_n).reshape(-1, 1)
        ELBO = - log_det_A - quadratic_term - ((K_nn.diagonal() - Q.diagonal()) / rho_diag).sum()

        return 0.5 * ELBO.squeeze()


class RRPVariationalELBOdiff(MarginalLogLikelihood):
    
    def __init__(self, 
            likelihood, 
            model,):
        
        super().__init__(likelihood, model)

    def forward(
            self,
            q):

        inducing_points = self.model.inducing_points
        num_inducing = inducing_points.size(-2)
        
        full_covar = q.covariance_matrix
        full_mean = q.mean.flatten()

        # Partition the kernel matrix u and f into blocks
        K_nn = full_covar[num_inducing:, num_inducing:]
        K_nz = full_covar[num_inducing:, :num_inducing]
        K_zz = full_covar[:num_inducing, :num_inducing]
        m_n = full_mean[num_inducing:]

        sigma_sq = self.likelihood.noise

        chol_K_zz = torch.linalg.cholesky(K_zz)
        K_zz_inv = cholesky_solve(torch.eye(chol_K_zz.size(-1), device=self.model.device), chol_K_zz)

        n_data = K_nz.size(0)
        Q = K_nz @ K_zz_inv @ K_nz.T
        
        D = torch.zeros(n_data, device=self.model.device)
        #S_comp =  list(set(range(n_data)) - set(self.model.S))
        
        for idx in range(len(self.model.rho)):
            D[idx] = self.model.rho[idx].noise
        
        D = torch.diag(D + sigma_sq)
        A = Q + D
        chol_A = torch.linalg.cholesky(A)
        A_inv = cholesky_solve(torch.eye(chol_A.size(-1), device=self.model.device), chol_A)
        
        expectation_term = self.model.train_targets - (A_inv @ (self.model.train_targets - m_n)).squeeze() / A_inv.diagonal().squeeze()
        var_term = 1 / A_inv.diagonal().squeeze()
        ELBO_diff =  1 / (torch.sqrt(K_nn.diagonal() - Q.diagonal())) * (self.model.train_targets - expectation_term) - var_term
        ELBO_diff = torch.clamp(ELBO_diff, min=0).sum()
       
        return ELBO_diff
