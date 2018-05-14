import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


print('loading onnx model')
onnx_model = onnx.load('train/model.onnx')

onnx.checker.check_model(onnx_model)

print('prepare tf model')
tf_rep = prepare(onnx_model)
print(tf_rep.predict_net)
print('-----')
print(tf_rep.predict_net.tensor_dict)

test = np.random.rand(1, 1, 28, 28)

out = tf_rep.run(test)._0
print(out)

with tf.Session() as persisted_sess:
    print("load graph")
    persisted_sess.graph.as_default()
    tf.import_graph_def(tf_rep.predict_net.graph.as_graph_def(), name='')
    # for op in persisted_sess.graph.get_operations():
    #    print(op)
    inp = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_input[0]].name
    )
    out = persisted_sess.graph.get_tensor_by_name(
        tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_output[0]].name
    )
    res = persisted_sess.run(out, {inp: test})
    print(res)

    tf.train.write_graph(persisted_sess.graph_def, "train", "test.pb", False)  # proto

tf_rep.export_graph('train/tf.pb')

