import numpy as np


class RegressionM: #construtor
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.beta = None
        
    def fit(self): #treinamento
        self.X = np.column_stack()