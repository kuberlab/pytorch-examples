import numpy as np
from caffe2.python import workspace

net = open('train/caffe2_net.pb').read()
init = open('train/caffe2_init.pb').read()

p = workspace.Predictor(init, net)

test = np.random.rand(1, 1, 28, 28).astype(np.float32)
print(p.run({'0': test}))
