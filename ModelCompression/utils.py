from pruning_utils import strip_pruning_vars
import tensorflow as tf # Default graph is initialized when the library is imported
import os
from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2


CHECKPOINT_DIR = 'checkpoint/Model1'
OUTPUT_NODE_NAME = 'add_5'
OUTPUT_DIR = 'checkpoint/Model1/PruneStrip'
OUTPUT_NAME = 'pruned_model1.pb'


CONTENT_DIR = 'test_img/content/'

INPUT_SHAPE = [1, None, None, 3]

import time


def strip():
    strip_pruning_vars.strip_pruning_vars(CHECKPOINT_DIR,OUTPUT_NODE_NAME, OUTPUT_DIR, OUTPUT_NAME)


def load_graph():
    filename = OUTPUT_DIR + '/' + OUTPUT_NAME

    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, INPUT_SHAPE, name="new_input")

        tf.import_graph_def(
            graph_def,
            name='prefix'
        )
        return graph



def save_sparse_weights():

    graph = load_graph()

    weights = {}
    for op in graph.get_operations():
        if 'masked_weight' in op.name:
            dense = graph.get_tensor_by_name(op.name + ":0")
            sparse = tf.contrib.layers.dense_to_sparse(dense)
            weights[op.name] = sparse
            print(type(dense))


    # print(type(weights[0]))
    # saver = tf.train.Saver(weights)
    # with tf.Session() as sess:
    #     saver.save(sess,OUTPUT_DIR)

def inference():
    pass
    # image = scipy.misc.imread(CONTENT_DIR + 'stata.jpg')

    # original_size = image.shape

    # image = scipy.misc.imresize(image, (256,256, 3))
    # input_im = np.zeros(shape=((1,) + image.shape), dtype=float)
    # input_im[0] = image
    # print(input_im.shape)

    # with tf.Session(graph=graph) as sess:

    #     summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    #     summary_op = tf.summary.merge_all()

    #     pruned_time = time.time()
    #     output = sess.run(y,feed_dict = {x : input_im})
    #     print("RUNTIME: {0}".format(time.time() - pruned_time))
    #     scipy.misc.imshow(output[0])


if __name__ == '__main__':
    save_sparse_weights()
    # strip()