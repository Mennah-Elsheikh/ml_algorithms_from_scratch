import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        # Learning rate for gradient descent
        self.lr = lr
        # Number of iterations for training
        self.n_iters = n_iters 
        # Weights and bias (initialized during fit)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Predict target values for samples in X
        return np.dot(X, self.weights) + self.bias