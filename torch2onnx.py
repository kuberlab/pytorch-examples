from __future__ import print_function

import argparse
import importlib

import torch.onnx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    required=True
)
parser.add_argument(
    '--output',
    required=True
)
parser.add_argument(
    '--model-class',
    help='path to model class. Example --model-class package1.module:ClassName',
    required=True
)
parser.add_argument(
    '--input-shape',
    required=True,
    nargs='+',
    help='comma-separated tensor dimensions. Example: --input-shape 1,1,28,28'
)
args = parser.parse_args()

module = ':'.join(args.model_class.split(':')[:-1])
class_name = args.model_class.split(':')[-1]

module = importlib.import_module(module)
clazz = getattr(module, class_name)
globals()[class_name] = clazz

print('loading torch model')
model = torch.load(args.input)

print('generating input shapes...')
inputs = ()
for input_shape in args.input_shape:
    shape = [int(i) for i in input_shape.split(',')]
    print('add shape %s' % shape)
    _input = torch.Tensor(*shape)
    inputs += (_input,)

print('exporting onnx model')
# Export to ONNX
model.train(False)
torch_out = torch.onnx._export(model, inputs, args.output, export_params=True)
