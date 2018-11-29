import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pruned_transform as transform
from layers.LookupConvolution2d import extract_dense_weights


MODEL_PATH = 'saved_model/lcnnModel1/model.ckpt-123402'

def inference():
    # loading test image
    im = scipy.misc.imread('test_img/content/stata.jpg')
    input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
    input_im[0] = im


    g1 = tf.Graph()

    with g1.as_default() as g:
        x_input = tf.placeholder(tf.float32, shape=input_im.shape)
        raw_preds = transform.net(x_input/255.0)

    with tf.Session(graph=g1) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        extract_dense_weights(sess)


    tf.reset_default_graph()

    g2 = tf.Graph()
    with g2.as_default() as g:
        x_input = tf.placeholder(tf.float32, shape=input_im.shape)
        raw_preds = transform.net(x_input/255.0)
        preds = tf.clip_by_value(raw_preds, 0, 255)

    with tf.Session(graph=g2) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)


        res = sess.run(preds, feed_dict={x_input: input_im})
        res = res.astype(np.uint8)
        # print(np.max(res))
        plt.imshow(res[0])
        plt.show()





if __name__ == '__main__':
    inference()



