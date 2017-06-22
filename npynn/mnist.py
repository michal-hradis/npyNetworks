from __future__ import print_function
from collections import OrderedDict
import numpy as np
from layers import Prod, ReLU, L2Loss, Sigmoid
from nets import SGD, LinearNet, testNet
from time import sleep

data = np.load('../data/mnist.npy').astype(np.float32)

labels = data[:, 0:1].T
data = data[:, 1:].T / 256.0

print(data.shape, labels.shape)
net = LinearNet()

net.add(Prod(in_dim=data.shape[0], out_dim=32))
#net.add(Sigmoid())
#net.add(Prod(in_dim=32, out_dim=32))
net.add(Sigmoid())
net.add(Prod(in_dim=32, out_dim=1))
net.add(L2Loss())

#testNet(net, data[:, :32], labels[:, :32])

batchSize = 512
opt = SGD(net, lr=0.001)
for i in range(1000000):
    batchPerm = np.random.choice(data.shape[1], batchSize)
    loss = opt.step(data[:, batchPerm], labels[:, batchPerm])

    print('iteration: {:d}, loss: {:f}'.format(i, loss))

    #print(i, type(layer).__name__, np.average(np.absolute(data)))
