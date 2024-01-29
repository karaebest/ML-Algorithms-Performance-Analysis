# Import statements
import pandas as pd
import numpy as np

np.random.seed(19680801)

import warnings

warnings.filterwarnings('ignore')


# Define Linear Regression Class
class LinearRegression:
    def __init__(self, add_bias=True, D=5, l2_reg=0, base='L'):
        self.add_bias = add_bias
        self.D = D  # Number of bases
        self.l2_reg = l2_reg  # Regularization parameter lambda
        self.base = base
        pass

    def fit(self, x, y):
        # Compute design matrix phi using specified base functions
        if self.base == 'P':
            phi = self.polynomial_design_matrix(x)
        else:
            phi = self.linear_design_matrix(x)
        # Add bias
        if self.add_bias:
            N = x.shape[0]
            phi.insert(0, 'X0', np.ones(N))
        norm = phi.T @ phi
        reg = self.l2_reg * np.identity(norm.shape[0])
        self.w = np.linalg.inv(norm + reg) @ phi.T @ y  # Determine parameters by computing closed form solution
        self.w.columns = ['W1', 'W2']
        return self

    def predict(self, x):
        # Compute design matrix phi using specified base functions
        if self.base == 'P':
            phi = self.polynomial_design_matrix(x)
        else:
            phi = self.linear_design_matrix(x)
        # Add bias
        if self.add_bias:
            N = x.shape[0]
            phi.insert(0, 'X0', np.ones(N))
        yh = np.matmul(phi, self.w)  # predict the y values
        yh.columns = ['Y\'1', 'Y\'2']
        return yh

    # Computes the design matrix Phi using a Polynomial feature function
    def polynomial_design_matrix(self, x):
        poly = lambda x, k: np.power(x, k)
        phi = pd.DataFrame()
        for col in x.columns:
            for i in range(self.D):
                feat = poly(x.loc[:, col], i + 1)
                phi = pd.concat([phi, feat], 1)
        return phi

    # Return copy of design matrix with linear features
    def linear_design_matrix(self, x):
        return x.copy()
