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

# import pruned_transform as transform
# from layers.LookupConvolution2d import extract_dense_weights


# MODEL_PATH = 'saved_model/lcnnModel1/model.ckpt-123402'
_conv_sparse = tf.load_op_library('/home/aschunk3/tensorflow/bazel-bin/tensorflow/core/user_ops/libconv_sparse.so')
conv_op = _conv_sparse.custom_convolution

def get_input(name='stata.jpg'):
    im = scipy.misc.imread('test_img/content/{0}'.format(name))
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

    stride = [1, stride, stride, 1]
    net = conv_op(net, weights, strides=stride)
    net = _instance_norm(net, shift, scale)
    if relu:
        net = tf.nn.relu(net)
    return net

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

def sparse_net(inputs, weights, shifts, scales):
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

def inference(input_im, sparsity, conv_type='sparse', name='stata.jpg'):
    # loading test image
    
    variables = np.load('weights/{0}-sparse.npy'.format(sparsity))[0]
    weights = variables[0]

    # for weight in weights:
    #     print (np.count_nonzero(weight) / weight.size)
    # shifts and scales are used for instance norms
    shifts = variables[1]
    scales = variables[2]
  

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
            # for i in range(100):
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
    print("Warming up...")
    # input_im = get_input(name='001.jpg')
    
    variables = np.load('weights/95-sparse.npy')[0]
    weights = variables[0]

    # for weight in weights:
    #     print (np.count_nonzero(weight) / weight.size)
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


if __name__ == '__main__':
    # input_im = get_input('001.jpg')
    fnames = glob.glob("./test_img/content/*.jpg")
    pos_sparsity=[50, 70, 90, 95]

    for fname in fnames:
        im = scipy.misc.imread(fname)
        input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
        input_im[0] = im
        dense_res = inference(input_im, 50, conv_type='dense')
        scipy.misc.imsave(fname.replace('content/', 'output/dense'), dense_res)
        for sparsity in pos_sparsity:
            sparse_res = inference(input_im, sparsity, conv_type='hybrid')
            scipy.misc.imsave(fname.replace('content/', 'output/{0}-'.format(sparsity)), sparse_res)




    # input_im = np.random.randn(1,1920,1080,3)
    # warmup(input_im)
    
    # for sparsity in pos_sparsity:
    #     inference(input_im, sparsity, conv_type='hybrid')


    # inference(input_im, 50, conv_type='dense')




    
    
        
    
    
    




