import onnx
import onnx_caffe2.backend

# Load the ONNX ModelProto object. model is a standard Python protobuf object
print('load onnx')
model = onnx.load("train/model.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
print('prepare caffe2')
prepared_backend = onnx_caffe2.backend.prepare(model)

prepared_backend.run()