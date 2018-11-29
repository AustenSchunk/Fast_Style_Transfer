import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt

# import pruned_transform as transform
# from layers.LookupConvolution2d import extract_dense_weights


# MODEL_PATH = 'saved_model/lcnnModel1/model.ckpt-123402'


def get_input():
    im = scipy.misc.imread('test_img/content/014.jpg')
    input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
    input_im[0] = im
    return input_im

def dense_conv(net, weights, stride, shift, scale, padding='SAME', relu=True):
    stride = [1, stride, stride, 1]
    net = tf.nn.conv2d(net, weights, stride, padding=padding)
    net = _instance_norm(net, shift, scale)
    if relu:
        net = tf.nn.relu(net)
    return net


def sparse_conv(net, weights, stride, shift, scale, padding='SAME', relu=True):
    ## TODO: replace regular conv with sparse conv
    pass

def conv_transpose(net, weights, stride, shift, scale):
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * stride), int(cols * stride)

    num_filters = weights.shape[2]
    # print (weights.shape)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,stride,stride,1]
    net = tf.nn.conv2d_transpose(net, weights, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net, shift, scale)
    return tf.nn.relu(net)


def _instance_norm(net, shift, scale):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    # shift = tf.Variable(tf.zeros(var_shape))
    # scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def dense_residual_block(net, weights, shifts, scales):
    tmp = dense_conv(net, weights[0], 1, shifts[0], scales[0])
    return net + dense_conv(tmp, weights[1], 1, shifts[1], scales[1], relu=False)

def sparse_residual_block(net, weights, shifts, scales):
    tmp = sparse_conv(net, weights[0], 1, shifts[0], scales[0])
    return net + sparse_conv(tmp, weights[1], 1, shifts[1], scales[1], relu=False)


def dense_net(inputs, weights, shifts, scales):
    conv1 = dense_conv(inputs, weights[0], 1, shifts[0], scales[0])
    conv2 = dense_conv(conv1,  weights[1], 2, shifts[1], scales[1])
    conv3 = dense_conv(conv2,  weights[2], 2, shifts[2], scales[2])
    resid1 = dense_residual_block(conv3, weights[3:5], shifts[3:5], scales[3:5])
    resid2 = dense_residual_block(resid1, weights[5:7], shifts[5:7], scales[5:7])
    resid3 = dense_residual_block(resid2, weights[7:9], shifts[7:9], scales[7:9])
    resid4 = dense_residual_block(resid3, weights[9:11], shifts[9:11], scales[9:11])
    resid5 = dense_residual_block(resid4, weights[11:13], shifts[11:13], scales[11:13])
    conv_t1 = conv_transpose(resid5, weights[13], 2, shifts[13], scales[13])
    conv_t2 = conv_transpose(conv_t1, weights[14], 2, shifts[14], scales[14])
    conv_t3 = dense_conv(conv_t2, weights[15], 1, shifts[15], scales[15], relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds





def inference(conv_type='dense'):
    # loading test image
    input_im = get_input()
    
    variables = np.load('weights/50-sparse.npy')[0]
    weights = variables[0]
    # shifts and scales are used for instance norms
    shifts = variables[1]
    scales = variables[2]


    if conv_type == 'dense':
        g1 = tf.Graph()

        with g1.as_default() as g:
            x_input = tf.placeholder(tf.float32, shape=input_im.shape)
            raw_preds = dense_net(x_input/255.0, weights, shifts, scales)
            preds = tf.clip_by_value(raw_preds, 0, 255)

        with tf.Session(graph=g1) as sess:
            # dense_start = time.time()
            res = sess.run(preds, feed_dict={x_input : input_im})
            res = res.astype(np.uint8)
            print(np.max(res))
            print(np.min(res))

            plt.imshow(res[0])
            plt.show()
            # print(res.shape)




    # g1 = tf.Graph()

    # with g1.as_default() as g:
    #     x_input = tf.placeholder(tf.float32, shape=input_im.shape)
    #     raw_preds = transform.net(x_input/255.0)

    # with tf.Session(graph=g1) as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, MODEL_PATH)
    #     extract_dense_weights(sess)


    # tf.reset_default_graph()

    # g2 = tf.Graph()
    # with g2.as_default() as g:
    #     x_input = tf.placeholder(tf.float32, shape=input_im.shape)
    #     raw_preds = transform.net(x_input/255.0)
    #     preds = tf.clip_by_value(raw_preds, 0, 255)

    # with tf.Session(graph=g2) as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, MODEL_PATH)


    #     res = sess.run(preds, feed_dict={x_input: input_im})
    #     res = res.astype(np.uint8)
    #     # print(np.max(res))
    #     plt.imshow(res[0])
    #     plt.show()





if __name__ == '__main__':
    inference()



