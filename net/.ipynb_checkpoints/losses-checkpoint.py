import numpy as np


''' loss functions '''

def l1(y, t, d=False):
    if d:
        return np.sign(y - t)
    return np.mean(np.abs(y - t), axis=-1)

def mse(y, t, d=False):
    if d:
        return 2 * (y - t) / y.shape[-1]
    return np.mean(np.square(y - t), axis=-1)

def bce(y, t, d=False):
    if d:
        return t / (1-y) - (1-t) / y
    return np.mean(-t*np.log(1-y) - (1-t)*np.log(y), axis=-1)

