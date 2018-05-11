import onnx
from onnx_tf.backend import prepare

print('loading onnx model')
onnx_model = onnx.load('train/model.onnx')

__import__('pdb').set_trace()
print('prepare tf model')
tf_rep = prepare(onnx_model)

print(tf_rep.predict_net)
print('-----')
print(tf_rep.input_dict)
print('-----')
print(tf_rep.uninitialized)

