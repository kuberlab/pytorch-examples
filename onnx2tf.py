import onnx
from onnx_tf.backend import prepare

print('loading onnx model')
onnx_model = onnx.load('train/model.onnx')

print('prepare tf model')
tf_rep = prepare(onnx_model)
print(tf_rep.predict_net)
print('-----')
print(tf_rep.predict_net.tensor_dict)
tf_rep.export_graph('train/tf.pb')
