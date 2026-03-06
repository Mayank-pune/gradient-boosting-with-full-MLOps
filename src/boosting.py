import numpy as np 
from src.tree import DecisionTree
from src.losses import SquaredLoss, LogLoss, sigmoid

class GradientBoosting:
    def __init__(self, n_trees, max_depth, learning_rate, loss):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        if(loss == "squared"):
            self.loss = SquaredLoss()
        elif(loss == "log"):
            self.loss = LogLoss() 
        self.trees = []

    def fit(self, X, y):
        self.initial_pred = self.loss.initial_prediction(y)
        y_pred = np.full(len(y), self.initial_pred)
        
        for i in range(self.n_trees):
            residuals = -self.loss.gradient(y, y_pred)
            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
    
    def predict(self, X):
        ini_pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            ini_pred+= self.learning_rate * tree.predict(X)
        return ini_pred
    
    def predict_proba(self, X):
        pred_raw_scores = self.predict(X)
        return sigmoid(pred_raw_scores)