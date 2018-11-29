from pruning_utils import strip_pruning_vars
import tensorflow as tf # Default graph is initialized when the library is imported
import os
from PIL import Image
import numpy as np
import scipy
import scipy.signal
from scipy import misc
import matplotlib.pyplot as plt
import cv2
import unittest

CHECKPOINT_DIR = 'checkpoint/Model1'
OUTPUT_NODE_NAME = 'add_5'
OUTPUT_DIR = 'checkpoint/Model1/PruneStrip'
OUTPUT_NAME = 'pruned_model1.pb'


CONTENT_DIR = 'test_img/content/'

INPUT_SHAPE = [1, None, None, 3]

import time

# sparse_conv2d_m = tf.load_op_library('/home/aschunk3/tensorflow/bazel-bin/tensorflow/core/user_ops/sparse_conv2d.so')


# def strip():
#     strip_pruning_vars.strip_pruning_vars(CHECKPOINT_DIR,OUTPUT_NODE_NAME, OUTPUT_DIR, OUTPUT_NAME)


def load_graph():
    filename = 'saved_model' + '/' + OUTPUT_NAME

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

def convert_sparse_weights(sparse_tensor, indices, values, shapes):
    indices.append(sparse_tensor.indices)
    values.append(sparse_tensor.values)
    shapes.append(sparse_tensor.dense_shape)

def save_sparse_weights():

    graph = load_graph()

    indices = []
    values = []
    shapes = []
    for op in graph.get_operations():
        if 'masked_weight' in op.name:
            dense = graph.get_tensor_by_name(op.name + ":0")
            sparse_tensor = tf.contrib.layers.dense_to_sparse(dense)
            convert_sparse_weights(sparse_tensor, indices, values, shapes)
            # weights[op.name] = sparse
            # print(type(dense))

    # return indices, values, shapes
    with tf.Session(graph=graph) as sess:
        res = sess.run([indices, values, shapes])
        np.save('sparse_weights.npy', res)
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

# class InnerProductOpTest(unittest.TestCase):

    

#     def test_sparse_conv2d(self):
#         # config = tf.ConfigProto(
#         #     device_count = {'GPU': 0}
#         # )
#         with tf.Session('') as sess:
#             x = tf.placeholder(tf.float32, shape=(1, 228, 228, 3))
#             conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

#             self.assertListEqual([1, 55, 55, 96], conv.get_shape().as_list())

#     def test_sparse_conv2d_simple(self):
#         # config = tf.ConfigProto(
#         #     device_count = {'GPU': 0}
#         # )
#         with tf.Session('') as sess:
#             x = tf.placeholder(tf.float32, shape=(1, 11, 11, 3))
#             conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

#             inp = np.zeros((1, 11, 11, 3))
#             out = sess.run(conv, feed_dict={x: inp})

#             self.assertEqual(0, np.count_nonzero(out))

#             inp[0][1][1][0] = 1
#             out = sess.run(conv, feed_dict={x: inp})
#             self.assertEqual(0, np.count_nonzero(out))

#             inp[0][0][0][0] = 1
#             out = sess.run(conv, feed_dict={x: inp})
#             self.assertEqual(1, np.count_nonzero(out))

    # def test_style_sparse(self):

    #     im = scipy.misc.imread('test_img/content/001.jpg')
    #     input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
    #     input_im[0] = im

    #     indices, values, shapes = np.load('sparse_weights.npy')

    #     ind = indices[0]
    #     val = values[0]
    #     shape = shapes[0].tolist()

    #     config = tf.ConfigProto(
    #         device_count = {'GPU': 0}
    #     )
    #     with tf.Session(config=config) as sess:
    #         x = tf.placeholder(tf.float32, shape=input_im.shape)


    #         sparse_conv = sparse_conv2d_m.sparse_conv2d(x, ind, val, shape, strides=[1, 1])

    #         dense_weights = np.zeros(shape=shape)
    #         for i in range(ind.shape[0]):
    #             dense_weights[tuple(ind[i])] = val[i]

    #         dense_conv = tf.nn.conv2d(x, dense_weights, [1,1,1,1], padding='VALID')


    #         sparse_start = time.time()
    #         sparse_out = sess.run(sparse_conv, feed_dict={x:input_im})
    #         print('Sparse Compute Time: {0}'.format(time.time() - sparse_start))

    #         dense_start = time.time()
    #         dense_out = sess.run(dense_conv, feed_dict={x:input_im})
    #         print('Dense Compute Time: {0}'.format(time.time() - dense_start))

    #         assert dense_out.shape == sparse_out.shape
    #         print('Total Difference: {0}'.format(np.sum(dense_out - sparse_out)))


def simple_sparse_conv(net, ind, val, shape, stride):

    img2col = tf.extract_image_patches(net, [1, shape[0], shape[1], 1],
                        [1, stride[0], stride[1], 1], [1, 1, 1, 1], 'VALID')
    img2col = tf.transpose(img2col, [0, 3, 1, 2])
    img2col_shape = img2col.get_shape().as_list()
    img2col = tf.reshape(img2col, [img2col_shape[1], img2col_shape[2] * img2col_shape[3]])

    # sparse kernel & bias
    sparse_kernel = tf.SparseTensor(ind, val, shape)
    print('\n\n\n\n\n\n\n\n\n')
    print(sparse_kernel.get_shape().as_list())

    # multiplication
    matmul = tf.sparse_tensor_dense_matmul(sparse_kernel, img2col)
    matmul = tf.transpose(matmul)
    output = tf.reshape(matmul, [1, img2col_shape[2], img2col_shape[3], dense_kernel_shp[0]])
    return output


class SparseConvTest(unittest.TestCase):



    def test_sparse_convolution(self):
        im = scipy.misc.imread('test_img/content/001.jpg')
        input_im = np.zeros(shape=((1,) + im.shape), dtype=float)
        input_im[0] = im

        indices, values, shapes = np.load('sparse_weights.npy')

        ind = indices[0]
        val = values[0]
        shape = shapes[0].tolist()
        dense_weights = np.zeros(shape=shape)
        for i in range(ind.shape[0]):
            dense_weights[tuple(ind[i])] = val[i]

        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=input_im.shape)
            # dense_conv = tf.nn.conv2d(x, dense_weights, [1,1,1,1], padding='VALID')
            # dense_out = sess.run(dense_conv, feed_dict={x:input_im})
            res = simple_sparse_conv(x, ind, val, shape, stride=[1,1])

            actual_res = sess.run(res)
            
        






if __name__ == '__main__':
    unittest.main()
    # strip()
    # save_sparse_weights()




