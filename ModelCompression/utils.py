"""
Strips pruning variables and saves sparse weights
"""


from pruning_utils import strip_pruning_vars
import tensorflow as tf
import numpy as np


# Global variables
CHECKPOINT_DIR = 'checkpoint/Model1'
OUTPUT_NODE_NAME = 'add_5'
OUTPUT_DIR = 'saved_model'
OUTPUT_NAME = 'pruned_model5.pb'

CONTENT_DIR = 'test_img/content/'
INPUT_SHAPE = [1, None, None, 3]


def strip():
    """
    Removes variables used in pruning from graph for model
    """
    strip_pruning_vars.strip_pruning_vars(CHECKPOINT_DIR,OUTPUT_NODE_NAME, OUTPUT_DIR, OUTPUT_NAME)

def load_graph():
    """
    Loads previously trained graph in order to extract sparse weights
    """
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
    """
    Converts weights to sparse version

    Args:
        sparse_tensor: tf.SparseTensor contain indices, values, and shape
        indices: list of numpy array corresponding to indices of each layer's weights
        values: list of numpy array corresponding to values of each layer's weights
        shapes: list of numpy array corresponding to shape of each layer's weights

    Returns:
        None
    """
    indices.append(sparse_tensor.indices)
    values.append(sparse_tensor.values)
    shapes.append(sparse_tensor.dense_shape)

def save_sparse_weights():
    """
    Saves sparse weights
    """

    graph = load_graph()

    weights = []
    shifts = []
    scales = []
    for op in graph.get_operations():


        if 'masked_weight' in op.name:
            dense = graph.get_tensor_by_name(op.name + ":0")
            weights.append(dense)
        if 'Variable' in op.name and not 'read' in op.name:
            if 'Variable_1' in op.name:
                scale = graph.get_tensor_by_name(op.name + ":0")
                scales.append(scale)
            else:
                shift = graph.get_tensor_by_name(op.name + ":0")
                shifts.append(shift)
    variables = [weights, shifts, scales]

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(graph=graph, config=config) as sess:
        res = sess.run([variables])
        np.save('weights/97-sparse.npy', res)

# if __name__ == '__main__':
#     strip()
#     save_sparse_weights()




