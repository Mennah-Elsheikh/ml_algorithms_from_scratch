import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Regularization parameter
        self.lambda_param = lambda_param
        # Learning rate for gradient descent
        self.lr = learning_rate
        # Number of iterations for training
        self.n_iters = n_iters
        # Weights and bias (initialized during fit)
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1 or 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent for n_iters
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the sample is correctly classified with margin
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Regularization and misclassification
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Predict class labels for samples in X
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)