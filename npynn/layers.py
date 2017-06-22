from __future__ import print_function
from collections import OrderedDict
import numpy as np


dataType = np.float64


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

        #init

    def forward(self, bottom):
        self.bottom = bottom
        return ##### TOTO

    def backward(self, top_grad):
        ##### TOTO
        return gradDown

    def getParams(self):
        return {self.name + '.W': (self.W, self.dW),
                self.name + '.b': (self.b, self.db)}


class ReLU(object):

    def forward(self, bottom):
        self.bottom = bottom
        return ##### TOTO

    def backward(self, grad):
        return ##### TOTO

    def getParams(self):
        return {}

class Sigmoid(object):

    def forward(self, bottom):
        return ##### TOTO

    def backward(self, grad):

        return ##### TOTO

    def getParams(self):
        return {}


class L2Loss(object):

    def forward(self, bottom, labels):
        self.labels = labels
        self.bottom = bottom
        return ##### TOTO

    def backward(self, grad):
        return ##### TOTO

    def getParams(self):
        return {}


class SoftmaxWithLoss(object):
    def init(self):
        pass

    def forward(self, bottom, labels):
        return ##### TOTO

    def backward(grad):
        return ##### TOTO

    def getParams(self):
        return {}


class Softmax(object):

    def forward(self, bottom):
        return ##### TOTO

    def backward(self, bottom, grad):
        pass

    def getParams(self):
        return {}
