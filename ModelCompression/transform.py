import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
import numpy as np
import re



TOWER_NAME = 'tower'
WEIGHTS_INIT_STDEV = .1
BATCH_SIZE = (1, 256, 256, 3)

def net(image):
    """
    Transforms image into stylized version using same architecture 
    as in Lengstrom's paper but applies thresholded pruning to the weights
    Args:
        image: image to be transformed
    Returns:
        stylized version of input image
    """
    conv1 = pruned_conv_layer(image, 32, 9, 1, 1)
    conv2 = pruned_conv_layer(conv1, 64, 3, 2, 2)
    conv3 = pruned_conv_layer(conv2, 128, 3, 2, 3)
    resid1 = _residual_block(conv3, 4)
    resid2 = _residual_block(resid1, 6)
    resid3 = _residual_block(resid2, 8)
    resid4 = _residual_block(resid3, 10)
    resid5 = _residual_block(resid4, 12)
    conv_t1 = pruned_conv_tranpose_layer(resid5, 64, 3, 2, 14)
    conv_t2 = pruned_conv_tranpose_layer(conv_t1, 32, 3, 2, 15)
    conv_t3 = pruned_conv_layer(conv_t2, 3, 9, 1, 16, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds
    

def pruned_conv_layer(net, num_filters, filter_size, strides, layerNum, relu=True):
    """
    Adds a convolutional layer with to be pruned weights onto computational graph
    Args:
        net
    """
    with tf.variable_scope('conv{0}'.format(layerNum)) as scope:

        kernel = pruned_conv_weights_init(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, pruning.apply_mask(kernel, scope), strides_shape, padding='SAME')
        net = _instance_norm(net)
        if relu:
            net = tf.nn.relu(net)

        _activation_summary(net)
        return net

def pruned_conv_tranpose_layer(net, num_filters, filter_size, strides, layerNum):
    """
    Performs convolution transpose in order to get to original dimensions
    """
    with tf.variable_scope('conv{0}'.format(layerNum)) as scope:
        kernel = pruned_conv_weights_init(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        net = tf.nn.conv2d_transpose(net, pruning.apply_mask(kernel, scope), tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)
        return tf.nn.relu(net)

    

def pruned_conv_weights_init(net, out_channels, filter_size, transpose=False):
    """
    Initializes prunable weights
    Args:
        net: built network up to this layer
        out_channels: number of output channels
        filter_size: dimension of filter i.e. filter_size x filter_size
        transpose: whether or not this layer is a conv transpose layer
    Returns:

    """
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = _variable_with_weight_decay(
        'weights', weights_shape, stddev=WEIGHTS_INIT_STDEV, wd=0.0)
    return weights_init

def _instance_norm(net, train=True):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _residual_block(net, layerNum, filter_size=3):
    tmp = pruned_conv_layer(net, 128, filter_size, 1, layerNum)
    return net + pruned_conv_layer(tmp, 128, filter_size, 1, layerNum+1, relu=False)



def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(name, shape,
                            tf.truncated_normal_initializer(
                            stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

if __name__ == '__main__':
    image = np.random.randint(0, 256, size=BATCH_SIZE)
    X_content = tf.placeholder(tf.float32, shape=BATCH_SIZE, name="X_content")
    # net = pruned_conv_layer(X_content, 32, 9, 1, 1)
    res = inference(X_content)
    print(res.shape)









