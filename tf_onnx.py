import onnxmltools
import tensorflow as tf
import tf2onnx
import os
import sys

filename = sys.argv[-1]
# Replace this with your desired input TF model name
input_model_name = filename

# Replace this with your desired output ONNX model name
base = os.path.basename(filename)
output_onnx_model = os.path.splitext(base)[0]+(".onnx")

directory = os.path.dirname(os.path.abspath(input_model_name))

if not os.path.exists(directory):
	directory = os.readlink(directory)
	#print (directory)
	output_onnx_model = directory + "\\" + output_onnx_model
else:
	output_onnx_model = filename.split(".")[0]+".onnx"

with tf.Session() as sess:
    # Note: this is a simple example Tensorflow model
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["input:0"], output_names=["output:0"])
    onnx_model = onnx_graph.make_model(input_model_name)

# Save as protobuf

onnxmltools.utils.save_model(onnx_model, output_onnx_model)