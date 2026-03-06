import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class SquaredLoss:
    def loss(self, y, y_pred):
        return np.mean((y-y_pred)**2)
    def gradient(self, y, y_pred):
        return -(y-y_pred)
    def initial_prediction(self,y):
        return np.mean(y)

class LogLoss:
    def loss(self, y, y_pred):
        p = sigmoid(y_pred)
        return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    def gradient(self, y, y_pred):
        p = sigmoid(y_pred)
        return -(y-p)
    def initial_prediction(self, y):
        p = np.mean(y)
        return np.log(p / (1-p))