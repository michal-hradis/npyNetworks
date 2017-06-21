from __future__ import print_function
from collections import OrderedDict
import numpy as np


dataType = np.float32

   
class Prod(object):
    counter = 0

    def __init__(self, in_dim, out_dim, name=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        if not name:
            Prod.counter += 1
            name = 'fc_{:d}'.format(self.counter)

        self.name = name

        self.W = np.zeros((out_dim, in_dim), dtype=dataType)
        self.b = np.zeros((out_dim, 1), dtype=dataType)
        self.dW = np.zeros((out_dim, in_dim), dtype=dataType)
        self.db = np.zeros((out_dim, 1), dtype=dataType)

        self.W[...] = (np.random.rand(*self.W.shape) - 0.5) \
            / self.in_dim**0.5 * 2

    def forward(self, bottom):
        self.bottom = bottom
        return np.dot(self.W, bottom) + self.b

    def backward(self, top_grad):
        self.dW += np.dot(top_grad, self.bottom.T)
        self.db += np.sum(top_grad, axis=1, keepdims=True)
        gradDown = np.dot(self.W.T, top_grad)
        return gradDown

    def getParams(self):
        return {self.name + '.W': (self.W, self.dW),
                self.name + '.b': (self.b, self.db)}


class ReLU(object):

    def forward(self, bottom):
        self.bottom = bottom
        return np.maximum(bottom, 0)

    def backward(self, grad):
        gradDown = grad.copy()
        gradDown[self.bottom < 0] = 0
        return gradDown

    def getParams(self):
        return {}

class Sigmoid(object):

    def forward(self, bottom):
        self.res = 1.0 / (1.0 + np.exp(-bottom))
        return self.res

    def backward(self, grad):
        gradDown = (1.0 - self.res) * self.res * grad.copy()
        return gradDown

    def getParams(self):
        return {}
    

class L2Loss(object):

    def forward(self, bottom, labels):
        self.labels = labels
        self.bottom = bottom
        return 0.5 * np.sum((bottom - labels)**2)

    def backward(self, grad):
        return grad * (self.bottom - self.labels)

    def getParams(self):
        return {}


class SoftmaxWithLoss(object):
    def init(self):
        pass

    def forward(self, bottom, labels):
        e_x = np.exp(bottom - np.amax(bottom, axis=1, keepdims=True))
        prob = e_x / np.sum(e_x, axis=1, keepdims=True)
        loss = -np.average(prob)
        return loss

    def backward(grad):
        pass #TODO ==========================

    def getParams(self):
        return {}


class Softmax(object):

    def forward(self, bottom):
        e_x = np.exp(bottom - np.amax(bottom, axis=1, keepdims=True))
        out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return out

    def backward(self, bottom, grad):
        pass

    def getParams(self):
        return {}
