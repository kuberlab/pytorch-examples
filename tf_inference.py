import os
from os import path

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


name = "train/tf.pb"
images_dir = 'mnist-images'


files = [path.join(images_dir, p) for p in os.listdir(images_dir)]


with tf.Session() as persisted_sess:
    with gfile.FastGFile(name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # test = np.random.rand(1, 1, 28, 28).astype(np.float32).argmax()
    inp = persisted_sess.graph.get_tensor_by_name('0:0')
    out = persisted_sess.graph.get_tensor_by_name('LogSoftmax:0')

    for f_name in files:
        data = imageio.imread(f_name).reshape(1, 1, 28, 28)
        base_name = path.basename(f_name)

        feed_dict = {inp: data}

        classification = persisted_sess.run(out, feed_dict)

        print('Prediction of file %s: %s' % (base_name, classification.argmax()))
