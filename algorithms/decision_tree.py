import numpy as np 
from collections import Counter

def entropy(y):
    """Calculate the entropy of a label array.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The label array.

    Returns
    -------
    float
        The entropy of the label array.
    """
    if len(y) == 0:
        return 0.0
    hist = np.bincount(y)
    counts = hist[hist > 0]
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small constant to avoid log(0)

class Node: 
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value to split on
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Class label for leaf nodes

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # Number of features to consider when looking for the best split
        self.root = None

    def fit(self, X, y):
        """Build the decision tree classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        y : np.ndarray, shape (n_samples,)
            The target labels.
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        if self.n_features is None:
            self.n_features = self.n_features_
        else:
            self.n_features = min(self.n_features, self.n_features_)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_index], threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _information_gain(self, y, feature_column, threshold):
        # Parent entropy
        parent_entropy = entropy(y)

        # Generate split
        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        # Weighted average child entropy
        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        e_left, e_right = entropy(y[left_indices]), entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain is difference in entropy
        ig = parent_entropy - child_entropy
        return ig
    
    def _split(self, X_column, threshold):
        left_indices = np.argwhere(X_column < threshold).flatten()
        right_indices = np.argwhere(X_column >= threshold).flatten()
        return left_indices, right_indices
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
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
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)