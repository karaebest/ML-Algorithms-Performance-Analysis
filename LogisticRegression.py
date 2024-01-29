# Import statements
import numpy as np

np.random.seed(19680801)

import warnings

warnings.filterwarnings('ignore')


# Logistic Regression class definition
class LogisticRegression:
    def __init__(self, lr=0.05, max_iters=1e4, l2_reg=0, epsilon=1e-4, add_bias=True, verbose=True):
        # Initialize hyper parameters
        self.add_bias = add_bias  # Boolean indicating if bias should be added to the model
        self.lr = lr  # Learning rate
        self.max_iters = max_iters  # Maximum number of iterations to run gradient descent
        self.w = None  # Weights
        self.l2_reg = l2_reg  # L2 regularization strength
        self.epsilon = epsilon  # Stopping criterion for gradient descent
        self.verbose = verbose  # Boolean indicating if intermediate steps should be printed

    def fit(self, X, y):
        # Add bias term if specified
        if self.add_bias:
            N = X.shape[0]
            X = X.copy()
            X.insert(0, 'X0', np.ones(N))
        N, D = X.shape  # Get dimensions of input data
        self.w = np.zeros(D)  # Initialize weights to zero
        grad = np.inf  # Initialize gradient to infinity
        t = 0  # Initialize iteration counter to zero

        # Function to calculate the gradient of the logistic regression cost function
        def gradient(X, y):
            N, D = X.shape
            yh = self.logistic(X)  # predictions  size N
            grad = np.dot(yh - y, X) / N + self.l2_reg * self.w  # divide by N because cost is mean over N points
            return grad  # size D

        # Run gradient descent until termination criteria met
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient(X, y)
            self.w = self.w - self.lr * grad
            t += 1

        # Print results if verbose flag is set
        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(grad)}')
            print(f'the weight found: {self.w}')

        return self

    def predict(self, X):
        if self.add_bias:
            N = X.shape[0]
            X = X.copy()
            X.insert(0, 'X0', np.ones(N))
        yh = self.logistic(X)  # Predictions
        yh = yh.round(0).astype(int)  # Round to nearest integer and convert to integer
        return yh  # Return predictions

    def logistic(self, X):
        return 1. / (1 + np.exp(-(X @ self.w)))  # Logistic function
