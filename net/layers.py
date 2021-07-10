import numpy as np

from .activations import *


class Layer:
    ''' fully connected layer.
    forward pass by calling object. '''

    def __init__(self, num_inputs, num_outputs, activation=linear, bias=True, trainable=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation

        self.weights = np.random.normal(0, 1 / np.sqrt(self.num_inputs), size=(num_outputs, num_inputs)).astype(np.float32)
        self.biases = np.zeros(num_outputs).astype(np.float32)
        
        self.bias = bias
        self.trainable = trainable

    def __call__(self, inputs, d=False):
        return self.activation(self.weights @ inputs + self.biases, d=d)

    def __repr__(self):
        return f"Layer({self.num_inputs}, {self.num_outputs}, activation={self.activation})"