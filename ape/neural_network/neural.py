import numpy as np

from ape._typing import Array, Matrix


def sigmoid(x: Matrix):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: Matrix) -> Matrix:
    return sigmoid(x) * (1.0 - sigmoid(x))

class Layer:
    def __init__(self, n_out: int, n_in):
        self.n = n_out
        self.w : Matrix = 0.001 * np.random.randn(n_out, n_in)
        self.b : Array = np.zeros((n_out, 1))

    
    def forward_propagation(self, inputs: Matrix) -> tuple[Matrix, Matrix]:
        self.i = inputs
        _, self.s = inputs.shape
        self.z = self.w @ inputs + self.b
        self.a = sigmoid(self.z)
        return self.a, self.z

    def backward_propagation(self, w, d):  # input means w[2].T @ d[2]
        self.d = (w.T @ d) * sigmoid_derivative(self.z)
        self.dw = 1/self.s * (self.d @ self.i.T)
        self.db = 1/self.s * (self.d @ np.ones((self.s, 1)))
        return self.w, self.d
    
    def gradient_descent(self, alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db


class NeuralNetwork:
    def __init__(self, neurons: list[int]):
        self.layers: list[Layer] = []
        n_in = neurons[0]
        for n in neurons[1:]:
            layer = Layer(n_out=n, n_in=n_in)
            self.layers.append(layer)
            n_in = n

    def forward_propagation(self, inputs: Matrix) -> Matrix:
        a = inputs
        for layer in self.layers:
            a, _ = layer.forward_propagation(a)
        return a
    
    def cost(self, inputs: Matrix, outputs: Matrix):
        y_hat = self.forward_propagation(inputs)
        y = outputs
        cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
        return cost


    def train(self, inputs: Matrix, outputs: Matrix, alpha=0.01):
        a = inputs
        for layer in self.layers:
            a, _ = layer.forward_propagation(a)

        d = a - outputs
        w = np.identity(d.shape[0])
        for layer in reversed(self.layers):
            w, d = layer.backward_propagation(w, d)
            layer.gradient_descent(alpha)
    


