import torch
import numpy as np

class Outliers(object):
    def __init__(
            self,
            x_train,
            y_train,
            fraction=0.2):
        
        self.x = x_train
        self.y = y_train        
        self.fraction = fraction

        self.n_data = self.x.size(-2)
        self.outliers_size = round(self.n_data * self.frac)
        
        self.outliers_idx =  np.sort( np.random.choice(np.arange(2, self.n_data - 2), size=self.outliers_size, replace=False) )
        self.non_outliers_idx = np.delete(np.arange(self.n_data), self.outliers_idx)

        self.x_train = self.x[self.non_outliers_idx]
        self.y_train = self.y[self.non_outliers_idx]

        self.x_outliers = self.x[self.outliers_idx]
        self.y_initial_outliers = self.y[self.outliers_idx]

        self.y_std = torch.std(self.y)

    def contaminate_symmetric(self, low=3.0, high=9.0):
        lower_bound = low * self.y_std
        upper_bound = high * self.y_std
        noise = lower_bound + (upper_bound - lower_bound) * torch.rand(self.outliers_size)
        y_outliers = self.y_initial_outliers + noise

        return self.x_train, self.y_train, self.x_outliers, y_outliers