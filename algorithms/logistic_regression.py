import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000):
        # Learning rate for gradient descent
        self.lr = lr
        # Number of iterations for training
        self.num_iter = num_iter
        # Weights and bias (initialized during fit)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Fit the model to the data
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1) # Convert labels if needed for stability, though usually 0/1
        # Gradient descent
        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Predict class labels for samples in X
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))