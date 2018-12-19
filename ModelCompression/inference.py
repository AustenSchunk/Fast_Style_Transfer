"""
Uses sparse and dense convolution with pretrained weights to perform style transfer.
Also tests runtimes for different architectures.
"""


import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import scipy.misc

from tensorflow.python.client import timeline

# Custom convolution op for faster convolution on sparse weights
_conv_sparse = tf.load_op_library('/home/aschunk3/tensorflow/bazel-bin/tensorflow/core/user_ops/libconv_sparse.so')
conv_op = _conv_sparse.custom_convolution

def get_input(name='stata.jpg'):
    """
    Accesses input from given file

    Args:
        name: file name

    Returns:
        image matrix
    """
    im = scipy.misc.imread('test_img/content/{0}'.format(name))
    input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
    input_im[0] = im
    return input_im

def dense_conv(net, weights, stride, shift, scale, padding='SAME', relu=True):
    """
    Performs standard built-in tensorflow convolution

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        weights: saved weights
        stride: distance of strides
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization
        padding: padding parameter of convolution
        relu: boolean to determine whether or not to use relu activations

    Returns:
        Transformed 4D tensor 
    """
    stride = [1, stride, stride, 1]
    net = tf.nn.conv2d(net, weights, stride, padding=padding)
    net = _instance_norm(net, shift, scale)
    if relu:
        net = tf.nn.relu(net)
    return net

def sparse_conv(net, weights, stride, shift, scale, padding='SAME', relu=True):
    """
    Performs convolution using sparse op for faster convolutions

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        weights: saved weights
        stride: distance of strides
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization
        padding: padding parameter of convolution
        relu: boolean to determine whether or not to use relu activations

    Returns:
        Transformed 4D tensor 
    """
    stride = [1, stride, stride, 1]
    net = conv_op(net, weights, strides=stride)
    net = _instance_norm(net, shift, scale)
    if relu:
        net = tf.nn.relu(net)
    return net

def conv_transpose(net, weights, stride, shift, scale):
    """
    Performs standard built-in tensorflow convolution transpose

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        weights: saved weights to be loaded in
        stride: distance of strides
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        Transformed 4D tensor 
    """
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
    """
    Performs instance normalization on output of a layer

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        shift: learned amount of shifting for instance norm
        scale: learned amount of scaling for instance norm

    Returns:
        Normalized output of layer
    """
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    # shift = tf.Variable(tf.zeros(var_shape))
    # scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def dense_residual_block(net, weights, shifts, scales):
    """
    Residual block using built-in tensorflow convolutions

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        weights: saved weights
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        4D input transformed by res block
    """
    tmp = dense_conv(net, weights[0], 1, shifts[0], scales[0])
    return net + dense_conv(tmp, weights[1], 1, shifts[1], scales[1], relu=False)

def sparse_residual_block(net, weights, shifts, scales):
    """
    Residual block using custom sparse convolutions

    Args:
        net: 4D Input into this layer of shape (batch size, input width, input height, num filters)
        weights: saved weights
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        4D input transformed by res block
    """
    tmp = sparse_conv(net, weights[0], 1, shifts[0], scales[0])
    return net + sparse_conv(tmp, weights[1], 1, shifts[1], scales[1], relu=False)

def dense_net(inputs, weights, shifts, scales):
    """
    Performs image transformation of input image(s) using dense convolutions

    Args:
        net: 4D Input of shape (batch size, input width, input height, num filters)
        weights: saved weights
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        4D input transformed by dense network
    """
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

def sparse_net(inputs, weights, shifts, scales):
    """
    Performs image transformation of input image(s) using sparse convolutions

    Args:
        net: 4D Input of shape (batch size, input width, input height, num filters)
        weights: saved weights
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        4D input transformed by sparse network
    """
    conv1 = sparse_conv(inputs, weights[0], 1, shifts[0], scales[0])
    conv2 = sparse_conv(conv1,  weights[1], 2, shifts[1], scales[1])
    conv3 = sparse_conv(conv2,  weights[2], 2, shifts[2], scales[2])
    resid1 = sparse_residual_block(conv3, weights[3:5], shifts[3:5], scales[3:5])
    resid2 = sparse_residual_block(resid1, weights[5:7], shifts[5:7], scales[5:7])
    resid3 = sparse_residual_block(resid2, weights[7:9], shifts[7:9], scales[7:9])
    resid4 = sparse_residual_block(resid3, weights[9:11], shifts[9:11], scales[9:11])
    resid5 = sparse_residual_block(resid4, weights[11:13], shifts[11:13], scales[11:13])
    conv_t1 = conv_transpose(resid5, weights[13], 2, shifts[13], scales[13])
    conv_t2 = conv_transpose(conv_t1, weights[14], 2, shifts[14], scales[14])
    conv_t3 = sparse_conv(conv_t2, weights[15], 1, shifts[15], scales[15], relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def hybrid_net(inputs, weights, shifts, scales):
    """
    Performs image transformation of input image(s) using both dense and sparse convolutions

    Args:
        net: 4D Input of shape (batch size, input width, input height, num filters)
        weights: saved weights
        shift: shifting parameter used in instance normalization
        scale: scale parameter used in instance normalization

    Returns:
        4D input transformed by hybrid network
    """
    conv1 = sparse_conv(inputs, weights[0], 1, shifts[0], scales[0])
    conv2 = dense_conv(conv1,  weights[1], 2, shifts[1], scales[1])
    conv3 = dense_conv(conv2,  weights[2], 2, shifts[2], scales[2])
    resid1 = dense_residual_block(conv3, weights[3:5], shifts[3:5], scales[3:5])
    resid2 = dense_residual_block(resid1, weights[5:7], shifts[5:7], scales[5:7])
    resid3 = dense_residual_block(resid2, weights[7:9], shifts[7:9], scales[7:9])
    resid4 = dense_residual_block(resid3, weights[9:11], shifts[9:11], scales[9:11])
    resid5 = dense_residual_block(resid4, weights[11:13], shifts[11:13], scales[11:13])
    conv_t1 = conv_transpose(resid5, weights[13], 2, shifts[13], scales[13])
    conv_t2 = conv_transpose(conv_t1, weights[14], 2, shifts[14], scales[14])
    conv_t3 = sparse_conv(conv_t2, weights[15], 1, shifts[15], scales[15], relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def inference(input_im, sparsity, conv_type='sparse'):
    """
    Performs image transformation for a single input

    Args:
        input_im: input to be transformed
        sparsity: %sparsity of weights to be used (out of 100)
        conv_type: type of convolution to be used (dense, sparse, or hybrid)

    Returns:
        Transformed image
    """
    
    # loading weights
    variables = np.load('weights/{0}-sparse.npy'.format(sparsity))[0]
    weights = variables[0]

    # shifts and scales are used for instance norms
    shifts = variables[1]
    scales = variables[2]
  

    # This section was used for testing times 
    # Can uncomment, then look in browser to see layer activation times
    if conv_type == 'dense':
        g1 = tf.Graph()

        with g1.as_default() as g:
            x_input = tf.placeholder(tf.float32, shape=input_im.shape)
            raw_preds = dense_net(x_input/255.0, weights, shifts, scales)
            preds = tf.clip_by_value(raw_preds, 0, 255)



        with tf.Session(graph=g1, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:

            # dense_start = time.time()
            res = sess.run(preds, feed_dict={x_input : input_im})
            # return time.time() - dense_start

            # print("Dense time:",time.time() - dense_start, 'W/shape =',input_im.shape)


            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('timeline-dense.json', 'w') as f:
            #     f.write(ctf)
            res = res.astype(np.uint8)
            return res[0]

            # plt.imshow(res[0])
            # plt.show()
            # print(res.shape)
    elif conv_type == 'hybrid':
        g1 = tf.Graph()

        with g1.as_default() as g:
            x_input = tf.placeholder(tf.float32, shape=input_im.shape)
            raw_preds = hybrid_net(x_input/255.0, weights, shifts, scales)
            preds = tf.clip_by_value(raw_preds, 0, 255)

        with tf.Session(graph=g1, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:


            # hybrid_start = time.time()
            # for i in range(100):
            res = sess.run(preds, feed_dict={x_input : input_im})
            # return time.time() - hybrid_start
            # print("{0}% Hybrid time:".format(sparsity), time.time() - hybrid_start)

            res = res.astype(np.uint8)
            return res[0]

            # plt.imshow(res[0])
            # plt.show()
            # print(res.shape)

    else:
        g1 = tf.Graph()

        with g1.as_default() as g:
            x_input = tf.placeholder(tf.float32, shape=input_im.shape)
            raw_preds = sparse_net(x_input/255.0, weights, shifts, scales)
            preds = tf.clip_by_value(raw_preds, 0, 255)

        with tf.Session(graph=g1, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            sparse_start = time.time()

            res = sess.run(preds, feed_dict={x_input : input_im})
            return time.time() - sparse_start
            print("Sparse time:",time.time() - sparse_start, 'W/shape =',input_im.shape)

            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('timeline-sparse.json', 'w') as f:
            #     f.write(ctf)


            # res = res.astype(np.uint8)

            # plt.imshow(res[0])
            # plt.show()

def warmup(input_im):
    """
    Warms up gpu for consisten runtime calculations

    Args:
        input_im: 4D input used for warming up

    Returns:
        None
    """
    # print("Warming up...")
    # input_im = get_input(name='001.jpg')
    
    variables = np.load('weights/95-sparse.npy')[0]
    weights = variables[0]

    # shifts and scales are used for instance norms
    shifts = variables[1]
    scales = variables[2]
    g1 = tf.Graph()

    with g1.as_default() as g:
        x_input = tf.placeholder(tf.float32, shape=input_im.shape)
        raw_preds = hybrid_net(x_input/255.0, weights, shifts, scales)
        preds = tf.clip_by_value(raw_preds, 0, 255)

    with tf.Session(graph=g1, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
        for i in range(20):
            res = sess.run(preds, feed_dict={x_input : input_im})

def tester(sparsity):
    """
    Used to calculate runtimes for varying frame sizes

    Args:
        sparsity: Current sparsity to be tested

    Returns:
        None
    """
    print("Testing Sparsity: {0}".format(sparsity))
    res = np.zeros(shape=(2,168))

    warmup()
    for idx, i in enumerate(range(240, 1441, 10)):
        print("Dim:", i)
        input_im = np.random.rand(1,i,i,3)
        res[0][idx] = inference(input_im, sparsity, conv_type='dense')


    # print("Testing sparse")
    # warmup()
    # for idx, i in enumerate(range(240, 1441, 10)):
    #     if (sparsity == 90 and i < 900) or sparsity >= 95:
    #         print("Dim:", i)
    #         input_im = np.random.rand(1,i,i,3)
    #         res[1][idx] = inference(input_im, sparsity, conv_type='sparse')
    #     else:
    #         break

    print("Testing dense")
    warmup()
    for idx, i in enumerate(range(240, 1440, 10)):
        print("Dim:", i)
        input_im = np.random.rand(1,i,i,3)
        res[1][idx] = inference(input_im, sparsity, conv_type='hybrid')

    # np.save('test_results/sparsity-{0}-times.npy'.format(sparsity), res)

# Uncomment to run testing of runtimes
# if __name__ == '__main__':

#     fnames = glob.glob("./test_img/content/*.jpg")
#     pos_sparsity=[50, 70, 90, 95]

#     for fname in fnames:
#         im = scipy.misc.imread(fname)
#         input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
#         input_im[0] = im
#         dense_res = inference(input_im, 50, conv_type='dense')
#         scipy.misc.imsave(fname.replace('content/', 'output/dense'), dense_res)
#         for sparsity in pos_sparsity:
#             sparse_res = inference(input_im, sparsity, conv_type='hybrid')
#             scipy.misc.imsave(fname.replace('content/', 'output/{0}-'.format(sparsity)), sparse_res)





    
    
        
    
    
    




