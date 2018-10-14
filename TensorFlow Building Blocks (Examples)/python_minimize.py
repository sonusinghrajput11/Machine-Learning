import numpy as np
from scipy.misc import derivative
import random

class BasicPythonMinimization:
    def optimize(self, loss_fun, X_, epochs, learning_rate):
        for i in range(epochs):
            X_ = X_ - (learning_rate * (derivative(loss_fun, X_, dx=1e-6)))
            print("Interation {itr} --> X : {res}".format(itr = i, res = X_))    
        return X_
    
    def minimize(self, loss, epochs, learning_rate):
        X_initial = random.random()
        self.optimize(loss, X_initial, epochs, learning_rate) 


def loss(X):
    return (X*X) - (4*X) + 4
basicPythonMinimization = BasicPythonMinimization()
basicPythonMinimization.minimize(loss, 150, 0.1)