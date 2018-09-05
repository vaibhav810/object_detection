import tensorflow as tf
import os, argparse

from tensorflow.python.framework import graph_util


saver = tf.train.import_meta_graph('./model1_final.ckpt-5000.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./model1_final.ckpt-5000")

output_node_names="concat"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")  
)
output_graph="./vk_lat_v5_test.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()
