from __future__ import print_function
from collections import OrderedDict
import numpy as np
from layers import Prod, ReLU, L2Loss, Sigmoid
from nets import SGD, LinearNet
from time import sleep

data = np.loadtxt('/home/ihradis/projects/2017-06-20_simple_NN/data/abalone.data').astype(np.float32)

labels = data[:, -1:].T
data = data[:, :-1].T

net = LinearNet()

net.add(Prod(in_dim=data.shape[0], out_dim=128))
net.add(Sigmoid())
net.add(Prod(in_dim=128, out_dim=128))
net.add(Sigmoid())
net.add(Prod(in_dim=128, out_dim=1))
net.add(L2Loss())
#x = net.forward(data, labels) / data.shape[0]
#a = net.backward(1.0 / data.shape[0])

batchSize = 512
opt = SGD(net, lr=0.0001)
for i in range(10000):
    batchPerm = np.random.choice(data.shape[1], batchSize)
    loss = opt.step(data[:, batchPerm], labels[:, batchPerm])
    
    print('iteration: {:d}, loss: {:f}'.format(i, loss / batchSize))

    #print(i, type(layer).__name__, np.average(np.absolute(data)))
