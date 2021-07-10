import numpy as np


''' activation functions '''

def linear(x, d=False):
    if d:
        return np.ones(x.shape)
    return x

def sigmoid(x, d=False):
    ex = np.exp(-x)
    if d:
        return ex / np.square(1.0 + ex)
    return 1.0 / (1.0 + ex)

def tanh(x, d=False):
    if d:
        return 1.0 - np.square(np.tanh(x))
    return np.tanh(x)

def relu(x, d=False):
    if d:
        return np.heaviside(x, 0.0)
    return np.maximum(np.zeros(x.shape), x)

def softplus(x, d=False):
    if d:
        return sigmoid(x)
    return np.log(1.0 + np.exp(x))

def softmax(x, d=False):
    if d:
        raise NotImplementedError
    return np.exp(x) / np.sum(np.exp(x))


