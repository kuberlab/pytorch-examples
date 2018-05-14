import numpy as np
import onnx
from caffe2.python.onnx import backend as caffe2

# Load the ONNX ModelProto object. model is a standard Python protobuf object
print('load onnx')
model = onnx.load("train/model.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
print('prepare caffe2')
prepared_backend = caffe2.prepare(model)

# run the model in Caffe2
x = np.random.rand(1, 1, 28, 28).astype(np.float32)
# x = torch.randn(1, 1, 28, 28)
# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
print(model.graph.input[0].name)
W = {model.graph.input[0].name: x}

# Run the Caffe2 net:
print('run caffe2')
c2_out = prepared_backend.run(W)[0]
print(c2_out)

init_net_output = open('train/caffe2_init.pb', 'wb')
output = open('train/caffe2_net.pb', 'wb')

init_net, predict_net = caffe2.Caffe2Backend.onnx_graph_to_caffe2_net(model)
init_net_output.write(init_net.SerializeToString())
output.write(predict_net.SerializeToString())
