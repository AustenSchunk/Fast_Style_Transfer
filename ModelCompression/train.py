"""
Script for training style transfer network to have sparse weights
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
import transform


warnings.resetwarnings()

import vgg
import numpy as np
import scipy
import cv2
import sys
import functools


# Global Variables
DATA_PATH = '/home/aschunk3/Desktop/ComputerVisionaries/train_data'
STYLE_PATH = '/home/aschunk3/Desktop/ComputerVisionaries/styles/lionStyle.jpg'
SAVE_DIR = 'checkpoints/fns.ckpt'

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 3e1
TOTAL_VAR_WEIGHT = 2e2

BATCH_SIZE = 2
NUM_EPOCHS = 2

LEARNING_RATE=7e-4
DATA_SIZE = 123403

TARGET_SPARSITY = 0.97
SPARSITY_END = 500


def generate_image(path):
    """
    Reads and resizes a single image

    Args:
        path: path to all images

    Returns: 
        single randomly selected image

    """
    filenames = tf.train.string_input_producer(
    tf.train.match_filenames_once(path + "/*.jpg"))
    reader = tf.WholeFileReader()
    key, f_input = reader.read(filenames)
    raw_im = tf.image.decode_jpeg(f_input)
    resized_im = tf.image.resize_images(raw_im, [256, 256])
    return resized_im

def get_style_features(style_path=STYLE_PATH):
    """
    Generates the gram matrices for given layers in vgg net.
    Copied from lengstrom's implementation

    Args:
        style_path: path to the desired style image

    Returns:
        gram matrix / style features for a desired style image
    """
    
    style_target = scipy.ndimage.imread(style_path)
    style_shape = (1,) + style_target.shape
    style_features = {}


    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    return style_features

def get_batch(batch_shape, sess, res):
    """
    Gets a single batch of images

    Args:
        batch_shape: shape of 4D batch tensor
        sess: current tf.Session
        res: variable to be run in sess to get an image

    Returns:
        batch of images
    """
    batch = np.zeros(shape=batch_shape)
    for i in range(batch_shape[0]):
        batch[i] = sess.run(res)
    return batch

def get_content_loss(X_content, X_pre):
    """
    Calculatest the loss between generated and content images

    Args:
        X_content: content image to be transformed
        X_pre: actual content image

    Returns:
        loss between content and generated images
    """

    content_features = {}
    content_net = vgg.net(X_pre)
    content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

    preds = transform.net(X_content/255.0)
    preds_pre = vgg.preprocess(preds)

    net = vgg.net(preds_pre)

    content_size = _tensor_size(content_features[CONTENT_LAYER]) * BATCH_SIZE
    assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
    raw_content_loss = 2 * tf.nn.l2_loss( net[CONTENT_LAYER] - content_features[CONTENT_LAYER])
    content_loss = CONTENT_WEIGHT * (raw_content_loss / content_size)

    return content_loss, net, preds

def get_style_loss(net, style_features):
    """
    Calculates loss between generated and style features

    Args:
        net: outputs of image transformation network
        style_features: features of style image

    Returns:
        Loss between generated and style features
    """

    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0,2,1])
        grams = tf.matmul(feats_T, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

    style_loss = STYLE_WEIGHT * functools.reduce(tf.add, style_losses) / BATCH_SIZE
    return style_loss

def get_tv_loss(preds, batch_shape):
    """
    Calculates Total Variational loss for denoising

    Args:
        preds: image transformed by stylization network
        batch_shape: shape of current batch

    Returns:
        Total variational loss
    """
    tv_y_size = _tensor_size(preds[:,1:,:,:])
    tv_x_size = _tensor_size(preds[:,:,1:,:])
    y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
    tv_loss = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / BATCH_SIZE

    return TOTAL_VAR_WEIGHT * tv_loss

def _tensor_size(tensor):
    """
    Calculates number of elements in given tensor

    Args:
        tensor: tensor to be used in size calucation

    Returns:
        number of elements in given tensor 
    """
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def initialize_pruning_op(global_step):
    """
    Initializes all variables needed for weight pruning and actual pruning op

    Args:
        global_step: global step parameter to be used in weighted decay threshold pruning

    Returns:
        pruning op
    """

    # Parse pruning hyperparameters
    override_params = {'target_sparsity' : TARGET_SPARSITY, 'sparsity_function_end_step':SPARSITY_END}
    pruning_hparams = pruning.get_pruning_hparams().override_from_dict(override_params)

    
    # Create a pruning object using the pruning hyperparameters
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

    # Use the pruning_obj to add ops to the training graph to update the masks
    # The conditional_mask_update_op will update the masks only when the
    # training step is in [begin_pruning_step, end_pruning_step] specified in
    # the pruning spec proto
    mask_update_op = pruning_obj.conditional_mask_update_op()

    # Use the pruning_obj to add summaries to the graph to track the sparsity
    # of each of the layers
    pruning_obj.add_pruning_summaries()

    return mask_update_op

def train():
    """
    Training method for a stylization network with pruned weights
    """

    # gram matrices for specified style layers
    style_features = get_style_features()

    batch_shape = (BATCH_SIZE, 256, 256, 3)

    # Actual training session
    with tf.Graph().as_default():

        res = generate_image(DATA_PATH)
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        # Creating global step needed for pruning op
        global_step = tf.train.get_or_create_global_step()

        # Setting up input
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # Computing all loss values
        content_loss, net, preds = get_content_loss(X_content, X_pre)
        style_loss = get_style_loss(net, style_features)
        tv_loss = get_tv_loss(preds, batch_shape)
        loss = content_loss + style_loss + tv_loss

        tf.summary.scalar('Content Loss', content_loss)
        tf.summary.scalar('Style Loss', style_loss)
        tf.summary.scalar('TV Loss', tv_loss)
        tf.summary.scalar('Total Loss', loss)

        # Training operation
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

        # Pruning operation
        pruning_op = initialize_pruning_op(global_step)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/')


        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir='checkpoint/',
            save_secs=None,
            save_steps=5000,
            saver=tf.train.Saver(),
            checkpoint_basename='model.ckpt',
            scaffold=None)


        with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as mon_sess:
            print("STARTING TRAINING")
            # initializing image generation
            mon_sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=mon_sess)

            for epoch in range(NUM_EPOCHS):
                
                print("Epoch: {0}".format(epoch))
                epoch_step = 1
                while epoch_step * BATCH_SIZE < DATA_SIZE:
                    glob_step = (epoch + 1) * epoch_step

                    # new input batch
                    X_batch = get_batch(batch_shape, mon_sess, res)
                    feed_dict = {X_content : X_batch}

                    to_get = [style_loss, content_loss, tv_loss, loss]

                    _, tup, summ = mon_sess.run([train_op, to_get, merged], feed_dict=feed_dict)

                    mon_sess.run(pruning_op)

                    if epoch_step % 500 == 0:
                        writer.add_summary(summ, global_step= glob_step // 500)

                    _style_loss,_content_loss,_tv_loss,_loss = tup

                    sys.stdout.write("Step {0} w/Total Loss: {1}\r".format(glob_step, _loss))
                    sys.stdout.flush()
                    epoch_step+=1

            coord.request_stop()
            coord.join(threads)

# if __name__ == '__main__':
#     train()


     