# Import statements
import pandas as pd
import numpy as np

np.random.seed(19680801)

import warnings

warnings.filterwarnings('ignore')


class MiniBatchSGD:

    def __init__(self, learning_rate=.1, max_iters=1e4, epsilon=1e-4, batch_size=1, fixed_lr=True):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.w_history = []  # To store the weight history for training curve visualization
        self.fixed_lr = fixed_lr

    def get_batches(self, x, y, batch_size):
        batches = []  # Array to store mini batches
        num_batches = x.shape[0] // batch_size  # Number of mini batches
        i = 0

        for i in range(num_batches):
            # Create mini batches
            mini_batch_x = x[i * batch_size: (i + 1) * batch_size]
            mini_batch_y = y[i * batch_size: (i + 1) * batch_size]
            batches.append((mini_batch_x, mini_batch_y))

        # If there is remainder data
        if x.shape[0] % batch_size != 0:
            mini_batch_x = x[i * batch_size: x.shape[0]]
            mini_batch_y = y[i * batch_size: x.shape[0]]
            batches.append((mini_batch_x, mini_batch_y))

        return batches

    def run(self, gradient_fn, x, y, w):
        grad = np.inf  # Initialize gradient
        t = 1  # Time step

        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:

            # Get mini batches
            batches = self.get_batches(x, y, self.batch_size)

            self.w_history.append(w)

            for batch in batches:
                # Compute the gradient with present weights w
                grad = gradient_fn(batch[0], batch[1], w)

                # Update the weights
                if self.fixed_lr:
                    w = w - self.learning_rate * grad
                else:
                    w = w - (t ** (-0.5)) * grad

            self.w_history.append(w)

            t += 1

        return w


class MiniBatchSGDMomentum:

    def __init__(self, learning_rate=.1, max_iters=1e4, epsilon=1e-8, batch_size=1, momentum=0.99, fixed_lr=True):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.momentum = momentum
        self.w_history = []  # To store the weight history for training curve visualization
        self.fixed_lr = fixed_lr

    def get_batches(self, x, y, batch_size):
        batches = []  # Array to store mini batches
        num_batches = x.shape[0] // batch_size  # Number of mini batches

        for i in range(num_batches + 1):
            # Create mini batches
            mini_batch_x = x[i * batch_size: (i + 1) * batch_size]
            mini_batch_y = y[i * batch_size: (i + 1) * batch_size]
            batches.append((mini_batch_x, mini_batch_y))

        # If there is remainder data
        if x.shape[0] % batch_size != 0:
            mini_batch_x = x[i * batch_size: x.shape[0]]
            mini_batch_y = y[i * batch_size: x.shape[0]]
            batches.append((mini_batch_x, mini_batch_y))

        return batches

    def run(self, gradient_fn, x, y, w):
        grad = np.inf  # Initialize gradient
        t = 1  # Time step

        change = 0  # For momentum calculation

        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:

            # Get mini batches
            batches = self.get_batches(x, y, self.batch_size)

            self.w_history.append(w)

            for batch in batches:
                # Compute the gradient with present weights w
                grad = gradient_fn(batch[0], batch[1], w)

                # Update the weights
                if self.fixed_lr:
                    new_change = self.learning_rate * grad + self.momentum * change
                else:
                    new_change = (t ** (-0.5)) * grad + self.momentum * change

                w = w - new_change
                change = new_change

            self.w_history.append(w)

            t += 1

        return w


class LogisticRegressionSGD:
    def __init__(self, l2_reg=0):
        self.w = None
        self.l2_reg = l2_reg

    def fit(self, x, y, optimizer):
        # Initialize parameters
        N = x.shape[0]
        x = x.copy()
        # Add bias
        x.insert(0, 'X0', np.ones(N))
        N, D = x.shape
        self.w = np.zeros(D)

        # gradient descent
        def gradient(x, y, w):
            N, D = x.shape
            yh = self.logistic(x)
            grad = np.dot(yh - y, x) / N + self.l2_reg * w
            return grad

        # Run the optimizer
        self.w = optimizer.run(gradient, x, y, self.w)

        return self

    def predict(self, x):
        N = x.shape[0]
        x = x.copy()
        # Add bias
        x.insert(0, 'X0', np.ones(N))

        yh = self.logistic(x)
        yh = yh.round(0).astype(int)
        return yh

    def logistic(self, x):
        return 1. / (1 + np.exp(-(x @ self.w)))


class LinearRegressionSGD:
    def __init__(self, D=5, l2_reg=0, base='L'):
        self.D = D  # Number of bases
        self.l2_reg = l2_reg  # Regularization parameter lambda
        self.base = base
        pass

    def fit(self, x, y, optimizer):
        # Compute design matrix phi using specified base functions
        if self.base == 'P':
            phi = self.polynomial_design_matrix(x)
        else:
            phi = self.linear_design_matrix(x)

        # Add bias
        N = x.shape[0]
        phi.insert(0, 'X0', np.ones(N))

        # Gradient descent
        def gradient(x, y, w):
            yh = x @ w
            N, D = x.shape
            yh = yh.to_numpy()
            y = y.to_numpy()
            x = x.to_numpy()

            grad = np.dot(x.T, yh - y) / N + self.l2_reg * w

            return grad

        w0 = np.zeros(phi.shape[1])  # Initialize the weights to 0

        self.w = optimizer.run(gradient, phi, y, w0)  # Run the optimizer to get the optimal weights
        self.w = pd.DataFrame(self.w, columns=['W'])

        return self

    def predict(self, x):
        # Compute design matrix phi using specified base functions
        if self.base == 'P':
            phi = self.polynomial_design_matrix(x)
        else:
            phi = self.linear_design_matrix(x)

        # Add bias
        N = x.shape[0]
        phi.insert(0, 'X0', np.ones(N))
        yh = np.matmul(phi, self.w)  # Predict the y values
        yh.columns = ['Y']

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
