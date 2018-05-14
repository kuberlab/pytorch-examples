import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


name = "train/tf.pb"

with tf.Session() as persisted_sess:
    print("load graph")
    with gfile.FastGFile(name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    test = np.random.rand(1, 1, 28, 28).astype(np.float32)

    inp = persisted_sess.graph.get_tensor_by_name('0:0')
    out = persisted_sess.graph.get_tensor_by_name('LogSoftmax:0')
    feed_dict = {inp: test}

    classification = persisted_sess.run(out, feed_dict)

