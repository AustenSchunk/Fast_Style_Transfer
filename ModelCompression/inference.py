import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pruned_transform as transform


MODEL_PATH = 'saved_model/lcnnModel1/model.ckpt-123402'

def inference():
    # loading test image
    im = scipy.misc.imread('test_img/content/001.jpg')
    input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
    input_im[0] = im



    with tf.Session() as sess:
        x_input = tf.placeholder(tf.float32, shape=input_im.shape)
        preds = transform.net(x_input/255.0)

        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)


        res = sess.run(preds, feed_dict={x_input: input_im})
        # print(np.max(res))
        plt.imshow(res[0])
        plt.show()





if __name__ == '__main__':
    inference()



