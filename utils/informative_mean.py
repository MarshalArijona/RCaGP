import torch
import gpytorch
import numpy as np
import time

'''
def simulate_expert_correction(x_outliers, 
                            objective, 
                            sigma_sq=1.0):
    y = objective(x_outliers)
    n_outliers = y.shape[0]
    noise = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(sigma_sq)), size=(n_outliers, 1))
    y_correct = y + noise
    return y_correct
'''

def simulate_expert_correction(y,  
                            sigma_sq=1.0):
    
    y = y.reshape(-1, 1)
    n_outliers = y.shape[0]
    device = y.device
    noise = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(sigma_sq)), size=(n_outliers, 1))
    noise = noise.to(device)
    y_correct = y + noise
    return y_correct
    
class InformativeMeanPrior(gpytorch.means.Mean):
    def __init__(self, 
                x_inliers,
                x_outliers,
                y_inliers,
                y_outliers,
                y_corrections,
                y_mean,
                y_std,
                sigma_sq_correction=1.0,
                const=0.0,
                top=3):
        
        super().__init__()
        self.x_inliers = x_inliers
        self.x_outliers = x_outliers
        self.y_inliers = y_inliers.reshape(-1)
        self.y_outliers = y_outliers.reshape(-1)
        self.y_mean = y_mean
        self.y_std = y_std
        self.y_corrections = y_corrections.reshape(-1)
        self.sigma_sq_correction = torch.tensor(sigma_sq_correction)
        self.const = torch.tensor(const)
        self.device = y_inliers.device
        
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(self.device)
        
        self.top = top
        self.jitter = 1e-6

        # get the posterior of the latent variables to compute the informative mean
        self.get_mean_posterior()
        self.avg_correction = self._get_avg_correction()  
    
    def _get_avg_correction(self, normalize=True):
        self.correction = self.expected_delta_bar.reshape(-1) * self.expected_mu.reshape(-1)
        avg_correction = torch.mean(self.correction)
        
        if normalize:
            avg_correction = (avg_correction - self.y_mean) / self.y_std
        
        return avg_correction.item()

    def local_mean(self):
        kernel_in_out = self.kernel(self.x_outliers, self.x_inliers)
        _, top_idx = torch.topk(kernel_in_out.evaluate(), k=self.top, dim=-1)
        y_neighbors = self.y_inliers[top_idx]        
        mean_y_neighbors = torch.mean(y_neighbors, dim=-1)
        prec_y_neighbors = 1 / (torch.var(y_neighbors, dim=-1) + self.jitter)
        return mean_y_neighbors, prec_y_neighbors
    
    def get_mean_posterior(self):
        #outlier-labels prior
        z_score = (self.y_outliers - self.y_mean) / self.y_std
        alpha_o = torch.absolute(z_score)
        beta_o = 1e-2

        #outlier-labels posterior
        post_alpha = alpha_o + 1
        post_beta = beta_o
        self.expected_delta_bar = post_alpha / (post_alpha + post_beta)

        #outlier-corrections prior
        mu_o, tau_o = self.local_mean()
        
        #outlier-corrections posterior
        self.expected_mu = (mu_o * tau_o + 1 / self.sigma_sq_correction * self.y_corrections) / (tau_o + 1 / self.sigma_sq_correction)

    def forward(
            self,
            X,
            normalize=True):
        
        '''
        if len(self.y_outliers) >= 2:
        
            #top_idx = len(self.y_outliers) // 2
            top_idx = 1
            kernel_in_out = self.kernel(X, self.x_outliers)
            _, top_idx = torch.topk(kernel_in_out.evaluate(), k=top_idx, dim=-1)
            mean_prior = torch.mean(self.correction[top_idx], dim=-1)

            if normalize:
                return (mean_prior.flatten() - self.y_mean) / self.y_std 

            return mean_prior.flatten()

        else:
        '''
        
        dim = X.size(-2)
        return torch.ones(dim).to(self.device) * self.avg_correction
    