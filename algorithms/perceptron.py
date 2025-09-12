import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        # Learning rate for weight updates
        self.lr = learning_rate
        # Number of iterations for training
        self.n_iters = n_iters
        # Activation function = unit step function
        self.activation_func = self._unit_step_func
        # Weights and bias (initialized during fit)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Fit the model to the data
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure labels are 0 or 1
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Training the model
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Linear output (dot product + bias)
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function
                y_predicted = self.activation_func(linear_output)

                # Calculate the update
                update = self.lr * (y_[idx] - y_predicted)

                # Update weights and bias
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # Predict class labels for samples in X
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply step function
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        # Unit step activation function: return 1 if x >= 0, else 0
        return np.where(x >= 0, 1, 0)