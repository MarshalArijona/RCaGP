import torch
import gpytorch
import numpy as np

def simulate_expert_correction(x_outliers, 
                            objective, 
                            sigma_sq=3.0):
    y = objective(x_outliers)
    n_outliers = y.shape[0]
    noise = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(sigma_sq)), size=(n_outliers, 1))
    y_correct = y + noise
    return y_correct

class InformativeMeanPrior(gpytorch.means.Mean):
    def __init__(self, 
                Y,
                outliers_idx,
                Y_corrections,
                Y_mean,
                Y_std,
                sigma_sq_correction=1.0,
                const=0.0,
                ):
        
        super().__init__()
        self.Y = Y.reshape(-1)
        self.y_mean = Y_mean
        self.y_std = Y_std
        self.outliers_idx = outliers_idx.numpy()
        self.Y_corrections = Y_corrections.reshape(-1)
        self.sigma_sq_correction = torch.tensor(sigma_sq_correction)
        self.const = torch.tensor(const)
        self.device = Y.device

        #get the posterior of the latent variables to compute the informative mean
        self.get_mean_posterior()
        self.avg_correction = self._get_avg_correction()  
    
    def _get_avg_correction(self):
        correction = self.expected_delta_bar.reshape(-1) * self.expected_mu.reshape(-1)
        avg_correction = torch.mean(correction)
        avg_correction = (avg_correction - self.y_mean) / self.y_std
        return (avg_correction + self.const).item()

    def local_mean(self, y, outliers_idx):
        jitter = 1e-6
        idx = outliers_idx
        len_idx = len(idx)
        idx = np.array([-2, -1, 1, 2]) + idx.reshape(-1, 1)
        idx = np.clip(idx, a_min=0, a_max=len_idx - 1)
        
        y_clone = y.clone().detach().reshape(-1)
        y_neighbors = y_clone[idx]

        mean = torch.mean(y_neighbors, dim=-1)
        precision = 1 / (torch.var(y_neighbors, dim=-1) + jitter)

        return mean, precision
    
    def get_mean_posterior(self):
        #outlier labels
        jitter=1e-6
        Y_outliers = self.Y[self.outliers_idx]
        Z_score = (Y_outliers - self.y_mean) / self.y_std
        alpha_o = torch.absolute(Z_score)
        beta_o = 1e-3

        post_alpha = alpha_o + 1
        post_beta = beta_o
        self.expected_delta_bar = post_alpha / (post_alpha + post_beta)

        #outlier corrections
        mu_o, tau_o = self.local_mean(self.Y, self.outliers_idx)
        self.expected_mu = (mu_o * tau_o + 1 / (self.sigma_sq_correction + jitter) * self.Y_corrections) / (tau_o + 1 / self.sigma_sq_correction)

    def forward(
            self, 
            X):

        '''
        K = self.kernel(X, self.x_outliers)
        n, d = K.shape

        _, top_k_indices = torch.topk(K.evaluate().to(X.device), self.k, dim=1)
        
        correction = self.expected_delta_bar.reshape(-1) * self.expected_mu.reshape(-1) * torch.ones((n, d)).to(X.device)
        topk_correction = correction[torch.arange(n)[:, None], top_k_indices]
        avg_correction = torch.mean(topk_correction, dim=-1)
        '''
        dim = X.shape[-2]
        return torch.ones(dim).to(self.device) * self.avg_correction