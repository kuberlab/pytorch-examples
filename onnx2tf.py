from os import path
import shutil

import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


print('loading onnx model')
onnx_model = onnx.load('train/model.onnx')
export_path = 'train/saved_model'

onnx.checker.check_model(onnx_model)

print('prepare tf model')
tf_rep = prepare(onnx_model)


def get_placeholder(tensor):
    input_tensor = tf.placeholder(
        dtype=tensor.dtype,
        shape=tensor.shape,
        name=tensor.name.split(':')[0]
    )
    return input_tensor


if path.exists(export_path):
    shutil.rmtree(export_path)


with tf.Session() as persisted_sess:
    print("load graph")
    persisted_sess.graph.as_default()
    tf.import_graph_def(tf_rep.predict_net.graph.as_graph_def(), name='')

    i_tensors = []
    o_tensors = []
    inputs = {}
    outputs = {}

    for i in tf_rep.predict_net.external_input:
        t = persisted_sess.graph.get_tensor_by_name(
            tf_rep.predict_net.tensor_dict[i].name
        )
        i_tensors.append(t)
        tensor_info = tf.saved_model.utils.build_tensor_info(t)
        inputs[t.name.split(':')[0].lower()] = tensor_info
        print(
            'input tensor [name=%s, type=%s, shape=%s]'
            % (t.name, t.dtype.name, t.shape.as_list())
        )
    print('')

    for i in tf_rep.predict_net.external_output:
        t = persisted_sess.graph.get_tensor_by_name(
            tf_rep.predict_net.tensor_dict[i].name
        )
        o_tensors.append(t)
        tensor_info = tf.saved_model.utils.build_tensor_info(t)
        outputs[t.name.split(':')[0]] = tensor_info
        print(
            'output tensor [name=%s, type=%s, shape=%s]'
            % (t.name, t.dtype.name, t.shape.as_list())
        )

    feed_dict = {}
    outs = []
    for i in i_tensors:
        feed_dict[i] = np.random.rand(*i.shape.as_list()).astype(i.dtype.name)

    print('test run:')
    res = persisted_sess.run(o_tensors, feed_dict=feed_dict)
    print(res)

    # print('INPUTS')
    # print(inputs)
    # print('OUTPUTS')
    # print(outputs)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        persisted_sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        })
    builder.save()
    print('Model saved to %s' % export_path)

# tf_rep.export_graph('train/tf.pb')

