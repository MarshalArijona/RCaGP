import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior

class SVGP(ApproximateGP):
    def __init__(self,
                mean_module,
                covar_module,
                inducing_points, 
                likelihood, 
                learn_inducing_locations=True):
        
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(SVGP, self).__init__(variational_strategy)
        
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.num_outputs = 1
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()
            self.likelihood.eval()
            fw = self(X)
            dist = self.likelihood(fw)

            return GPyTorchPosterior(dist)