import numpy as np 
from algorithms.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # Number of features to consider when looking for the best split
        self.trees = []

    def fit(self, X, y):
        """Build the random forest classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        y : np.ndarray, shape (n_samples,)
            The target labels.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            # Bootstrap sampling
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            The predicted class labels.
        """
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.array([np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])])