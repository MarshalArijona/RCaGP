import numpy as np
from sklearn.preprocessing import RobustScaler

class RobustScaler2(RobustScaler):

    def inverse_transform_mean(self, x):
        '''To transform mean'''
        return x*self.scale_ + self.center_

    def inverse_transform_std(self, x):
        '''To transform standard deviation'''
        return x*self.scale_

class MyScaler:
    def __init__(self):
        pass

    def fit(self, x):
        self.mu  = np.median(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x-self.mu)/self.std

    def inverse_transform_mean(self, x):
        '''To transform mean'''
        return x*self.std + self.mu

    def inverse_transform_std(self, x):
        '''To transform standard deviation'''
        return x*self.std