import torch
from gpytorch.variational import _VariationalStrategy

class RCGPVariationalStrategy(_VariationalStrategy):
    def __init__(
            self, 
            model, 
            inducing_points,
            train_x,
            train_y,
            weight_function, 
            learn_inducing_locations=True):
        
        super(RCGPVariationalStrategy, self).__init__(model,
                        inducing_points,
                        variational_distribution=None,
                        learn_inducing_locations=learn_inducing_locations,
                        )

        self.train_x = train_x
        self.train_y = train_y
        self.weight_function = weight_function

        # Compute the kernel matrix between data train and inducing points
        self.num_inducing = self.inducing_points.size(-2)
        self.f_inputs = torch.cat([self.inducing_points, self.train_x], dim=-2)
    
    def forward(
            self, 
            x,
        ):

        f_mean = self.model.mean_module(self.f_inputs)
        f_covar = self.model.covar_module(self.f_inputs)

        # Partition the kernel matrix u and f into blocks
        m_f = f_mean[self.num_inducing:]
        K_fu = f_covar[self.num_inducing:, :self.num_inducing]

        # Compute J_w and y - mw
        w = self.weight_function.W(self.train_x, self.train_y)
        sigma_sq = self.model.likelihood.noise
        J_w = sigma_sq / 2 * torch.diag(1 / w ** 2)
        m_w = m_f + self.weight_function.dylog2(self.train_x, self.train_y)
        y_min_mw = self.y_train - m_w

        # Compute the kernel matrix between data test and inducing points
        full_inputs = torch.cat([self.inducing_points, x], dim=-2)
        full_covar = self.model.covar_module(full_inputs)

        # Partition the kernel matrix u and s into blocks
        K_uu = full_covar[ :self.num_inducing, :self.num_inducing].add_jitter(1e-6)
        K_su = full_covar[ self.num_inducing:, :self.num_inducing]
        K_ss = full_covar[ self.num_inducing:, self.num_inducing:]

        # Compute predictive posterior
        predictive_mean, predictive_covar = self._approximate_posterior(
            K_uu, 
            K_ss,
            K_su,
            J_w,
            K_fu,
            sigma_sq,
            m_f,
            y_min_mw
        )
        return torch.distributions.MultivariateNormal(predictive_mean, predictive_covar)
    
    def _approximate_posterior(
            self,  
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
        K_uu_inv = torch.cholesky_solve(torch.eye(chol_factor.size(-1)), chol_factor)

        #Compute variational distribution mu_u and Sigma_u
        J_w_inv = torch.diag(1.0 / J_w.diagonal())
        P_u = K_uu + K_fu @ (1 / sigma_sq * J_w_inv) @ K_fu.T  
        P_u_factor = torch.linalg.cholesky(P_u)
        P_u_inv = torch.cholesky_solve(torch.eye(P_u_factor.size(-1), P_u_factor))
        mu_u = m_f + K_uu @ P_u_inv @ K_fu.T @ (1 / sigma_sq * J_w_inv) @ y_min_mw
        Sigma_u = K_uu @ P_u_inv @ K_uu

        #Compute mean posterior
        phi_u = K_uu_inv @ K_su.T
        predictive_mean = phi_u.T @ mu_u

        # Compute the covariance of the posterior
        predictive_covar = K_ss - phi_u.T @ (K_uu - Sigma_u) @ phi_u

        return predictive_mean, predictive_covar