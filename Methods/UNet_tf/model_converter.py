import os, argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))
from tensorflow.python.tools import freeze_graph

def freeze_(model_folder, output_nodes='y_hat',
                 output_filename='frozen-graph.pb',
                 rename_outputs=None):
    # Load checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    output_graph = output_filename

    # Devices should be cleared to allow Tensorflow to control placement of
    # graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    '''
    # https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o + ':0'), name=n)
            onames = nnames
    '''
    input_graph_def = graph.as_graph_def()

    '''
    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    '''
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def,
            onames  # unrelated nodes will be discarded
        )

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def model_freeze(pbtxt_filepath, ckpt_filepath, pb_filepath):
    freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False,
                              input_checkpoint=ckpt_filepath, output_node_names='cnn/output',
                              restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph=pb_filepath, clear_devices=True, initializer_nodes='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and freeze weights from checkpoints into production models')
    parser.add_argument("--pbtxt_filepath",
                        default='./LightCNN/models_rotate_contrast/UNet60000.pbtxt',
                        type=str, help="Path to pbtxt file")
    parser.add_argument("--ckpt_filepath",
                        default='./LightCNN/models_rotate_contrast/UNet.ckpt',
                        type=str, help="path of checkpoint file")
    parser.add_argument("--output_graph",
                        default='./LightCNN/models_rotate_contrast/UNet60000.pb',
                        type=str, help="Output graph filename")
    args = parser.parse_args()

    model_freeze(pbtxt_filepath = args.pbtxt_filepath, ckpt_filepath = args.ckpt_filepath, pb_filepath = args.output_graph)
    #freeze_graph(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs)