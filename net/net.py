import time
import numpy as np

from .activations import *
from .losses import *
from .layers import *


class Net:
    ''' layer based neural network.
    
    attributes:
        layers: a list of network layers
        lr: learning rate
        loss: loss function (use function 'l1' or 'mse')
    
    methods:
        backward(input, target):
            calculates gradients of specified loss function w.r.t. weights and biases of layers
        step(grad_w, grad_b):
            applies gradient descent (gradients can be obtained via 'backward' method
        train(data, targets, epochs, batch_size, multiplier):
            trains network end to end on specified data and targets.
            
    '''

    def __init__(self, layers=[], lr=1e-3, loss=mse):
        self.layers = layers
        self.lr = lr
        self.loss = loss

    def __call__(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return "Net({}, lr={}, loss={})".format(
            tuple([self.layers[0].num_inputs] + [layer.num_outputs for layer in self.layers]),
            self.lr,
            self.loss)
    
    def add(self, layer):
        self.layers.append(layer)

    def backward(self, inputs, target):
        '''get gradients for entire net from single (batched) forward pass.'''
        grad_w = []
        grad_b = []

        # forward pass inputs
        x = inputs
        o = [x]
        for layer in self.layers:
            x = layer(x)
            o.append(x)
        
        assert x.shape == target.shape, 'output shape must equal target shape.'
        
        # backprop
        delta = self.loss(x, target, d=True)
        for i, layer in reversed(list(enumerate(self.layers))):
            if i+1 == len(self.layers):
                delta = delta * layer(o[i], d=True)
            else:
                delta = np.einsum('ik,jk->ij', delta, self.layers[i+1].weights) * layer(o[i], d=True)
            
            dw = np.einsum('ij,ik->ijk', o[i], delta)
            db = delta

            grad_w.append(dw)
            grad_b.append(db)

        return list(reversed(grad_w)), list(reversed(grad_b))

    def step(self, grads_w, grads_b):
        '''apply gradients from single backward pass'''
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.weights -= self.lr * grads_w[i].sum(0)
                if layer.bias:
                    layer.biases -= self.lr * grads_b[i].sum(0)
                    
    def train(self, data, targets, epochs=3, batch_size=1, multiplier=1):
        print("training...\n")
        I = data.shape[0] // batch_size * multiplier
        
        t0 = time.time()
        losses_epochs = []
        losses_batches = []
        for epoch in range(epochs):
            print(f"epoch {epoch+1}/{epochs}")
            epoch_loss = []
            for i in range(I):                
                choices = np.random.randint(data.shape[0], size=(batch_size,))
                x = data[choices]
                t = targets[choices]

                self.step(*self.backward(x, t))
                
                batch_loss = self.loss(self(x), t).mean()
                epoch_loss.append(batch_loss)
                losses_batches.append(batch_loss)
                
                print("({:3.1f}%) [{}] - loss: {:.3f}/{:.3f}".format(
                    (i+1)/I*100,
                    "="*int(32*i/I) + ">" + "."*int(32*(I-i-1e-3)/I),
                    batch_loss,
                    np.mean(epoch_loss)
                ), end="\r")
                
            losses_epochs.append(np.mean(epoch_loss))
            print(" "*100, end="\r")
            print("({:3.1f}%) [{}] - loss: {:.3f}".format(
                100,
                "="*32,
                np.mean(epoch_loss)
            ))
        t1 = time.time()
        print("\ntraining finished.")
        return losses_batches, losses_epochs, t1-t0
