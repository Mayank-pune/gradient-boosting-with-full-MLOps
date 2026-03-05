import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    def fit(self,X,y):
        self.root = self._build_tree(X, y, depth=0)
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        if n_samples <= self.min_samples_split:
            return Node(value = np.mean(y))
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value = np.mean(y))
        if len(np.unique(y))==1:
            return Node(value = np.mean(y))
        
        best_feature, best_threshold = self._best_split(X,y)
        if(best_feature is None):
            return Node(value = np.mean(y))
        
        left_mask = X[:, best_feature] <= best_threshold         # left_mask is a boolean array
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        left_child = self._build_tree(X_left, y_left, depth+1)
        right_child = self._build_tree(X_right, y_right, depth+1)
        
        return Node(feature = best_feature, threshold = best_threshold, left = left_child, right = right_child)
        
    def _mse(self, y):
        return np.mean((y-np.mean(y))**2)
    def _best_split(self, X, y):
        best_mse_reduction = -1
        best_feature = None
        best_threshold = None
        
        parent_mse = self._mse(y)
        n_samples = len(y)
        
        n_features = X.shape[1]
        
        for features_index in range(n_features):
            feature_values = X[:, features_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                y_left = y[feature_values<=threshold]
                y_right = y[feature_values>threshold]
                if(len(y_left)==0 or len(y_right)==0):
                    continue
                weighted_child_mse = ((len(y_left)/n_samples)*(self._mse(y_left)))+((len(y_right)/n_samples)*(self._mse(y_right)))
                mse_reduction = parent_mse - weighted_child_mse
                if(mse_reduction > best_mse_reduction):
                    best_mse_reduction = mse_reduction
                    best_feature = features_index
                    best_threshold = threshold
        return best_feature, best_threshold
    def _traverse(self, x, node):
        if(node.value is not None):
            return node.value
        if(x[node.feature] <= node.threshold):
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

        
