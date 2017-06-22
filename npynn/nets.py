from __future__ import print_function
from collections import OrderedDict
import numpy as np

class SGD(object):

    def __init__(self, net, lr=0.001):
        self.lr = lr
        self.net = net

    def step(self, data, labels):
        netParams = self.net.getParams()
        for name, param in netParams.items():
            param[1][...] = 0

        losses = self.net.forward(data, labels)
        self.net.backward(1.0 / data.shape[0])

        for name, param in netParams.items():
            param[0][...] = param[0] - self.lr * param[1]

        return losses


class LinearNet(object):
    def __init__(self):
        self.layers = []
        self.results = []

    def add(self, layer):
        self.layers.append(layer)
        self.results.append(0)

    def forward(self, data, labels=None):
        allResults = []
        for i, layer in enumerate(self.layers[:-1]):
            data = layer.forward(data)
            self.results[i] = data
        if labels is not None:
            data = self.layers[-1].forward(data, labels)
            self.results[-1] = data
        else:
            data = self.layers[-1].forward(data)
            self.results[-1] = data
        return data

    def backward(self, gradient):
        for i, layer in enumerate(self.layers[::-1]):
            gradient = layer.backward(gradient)
            #print(len(self.layers) - i,
            #    type(layer).__name__, np.average(np.absolute(gradient)))

        return gradient

    def getParams(self):
        netParams = OrderedDict()
        for layer in self.layers:
            layerParams = layer.getParams()
            for paramName in layerParams:
                netParams[paramName] = layerParams[paramName]
        return netParams


def testNet(net, data, labels):
    loss = net.forward(data, labels)
    net.backward(1)
    params = net.getParams()
    epsilon = 1e-5
    paramsToTest = 100
    for name, param in params.items():
        flatParam = param[0].reshape(-1)
        flatGrad = param[1].reshape(-1)
        gradients = []
        gradDiffs = []
        for index in np.random.choice(flatParam.size, paramsToTest):
            tmp = flatParam[index]
            flatParam[index] += epsilon
            newLoss = net.forward(data, labels)
            numericalGradient = (newLoss - loss) / epsilon
            flatParam[index] = tmp
            gradients.append(flatGrad[index])
            gradDiffs.append(
                numericalGradient - flatGrad[index])

        print(
            name,
            np.average(np.absolute(gradDiffs)) / np.average(np.absolute(gradients)),
            np.absolute(gradDiffs).max() / np.absolute(gradients).mean())





