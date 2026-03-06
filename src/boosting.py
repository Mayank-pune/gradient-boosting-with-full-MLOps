import numpy as np
from src.tree import DecisionTree
from src.losses import SquaredLoss, LogLoss, sigmoid

class GradientBoosting:
    def __init__(self, n_trees, max_depth, learning_rate, loss, subsample=1.0, n_iter_no_change=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        if(loss == "squared"):
            self.loss = SquaredLoss()
        elif(loss == "log"):
            self.loss = LogLoss()
        self.trees = []
        self.subsample = subsample
        self.n_iter_no_change = n_iter_no_change

    def fit(self, X, y, X_val=None, y_val=None):
        self.initial_pred = self.loss.initial_prediction(y)
        y_pred = np.full(len(y), self.initial_pred)

        best_val_loss = float('inf')
        rounds_without_improvement = 0
        best_n_trees = 0

        for i in range(self.n_trees):
            residuals = -self.loss.gradient(y, y_pred)
            tree = DecisionTree(max_depth = self.max_depth)
            ## Subsampling 
            new_num_samples = int(self.subsample * len(X))
            rand_indices = np.random.choice(len(X), size = new_num_samples, replace = False)
            X_sampled = X[rand_indices]
            residuals_sampled = residuals[rand_indices]
            tree.fit(X_sampled, residuals_sampled)
            ## ------------------------
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            if self.n_iter_no_change is not None and X_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self.loss.loss(y_val, y_val_pred)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                    best_n_trees = len(self.trees)
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.n_iter_no_change:
                    self.trees = self.trees[:best_n_trees]
                    break

    def predict(self, X):
        ini_pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            ini_pred+= self.learning_rate * tree.predict(X)
        return ini_pred

    def predict_proba(self, X):
        pred_raw_scores = self.predict(X)
        return sigmoid(pred_raw_scores)
