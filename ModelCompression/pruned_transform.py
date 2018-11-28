import tensorflow as tf, pdb

from layers.LookupConvolution2d import lookup_conv2d


WEIGHTS_INIT_STDEV = .1

DICT_SIZES = [30, 120, 120, 240, 240, 240, 240, 512, 512]
LAMBDA = 0.3

def net(image):
    conv1 = _conv_layer(image, 32, 9, 1, layerNum=1, dict_size=DICT_SIZES[0])
    conv2 = _conv_layer(conv1, 64, 3, 2, layerNum=2, dict_size=DICT_SIZES[1])
    conv3 = _conv_layer(conv2, 128, 3, 2, layerNum=3, dict_size=DICT_SIZES[2])
    resid1 = _residual_block(conv3, filter_size=3, layerNum=4, dict_size=DICT_SIZES[3])
    resid2 = _residual_block(resid1, filter_size=3, layerNum=6, dict_size=DICT_SIZES[4])
    resid3 = _residual_block(resid2, filter_size=3, layerNum=8, dict_size=DICT_SIZES[5])
    resid4 = _residual_block(resid3, filter_size=3, layerNum=10, dict_size=DICT_SIZES[6])
    resid5 = _residual_block(resid4, filter_size=3, layerNum=12, dict_size=DICT_SIZES[7])
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, layerNum=14)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, layerNum=15)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1,  layerNum=16, relu=False, dict_size=DICT_SIZES[8])
    with tf.variable_scope('transform_output') as scope:
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
        return preds

def _conv_layer(net, num_filters, filter_size, strides, layerNum, relu=True, dict_size=240):
    # weights_init = _conv_init_vars(net, num_filters, filter_size)
    # strides_shape = [1, strides, strides, 1]
    # net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    with tf.variable_scope('Layer-{0}-conv'.format(layerNum)) as scope:
        kernel_size=[filter_size,filter_size]
        stride = [strides, strides]
        net = lookup_conv2d(tensor_in=net, num_outputs=num_filters, kernel_size=kernel_size , stride=stride, dict_size=dict_size)
        net = _instance_norm(net)
        if relu:
            net = tf.nn.relu(net)

        return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides, layerNum):
    with tf.variable_scope('Layer-{0}-convtranpose'.format(layerNum)) as scope:
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)
        return tf.nn.relu(net)

def _residual_block(net, layerNum, filter_size=3, dict_size=240):
    tmp = _conv_layer(net, 128, filter_size, 1, layerNum, dict_size=dict_size)
    return net + _conv_layer(tmp, 128, filter_size, 1, layerNum + 1, relu=False, dict_size=dict_size)

def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init