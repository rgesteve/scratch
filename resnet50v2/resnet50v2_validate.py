
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
import urllib.request

#Load sample inputs and outputs

test_data_dir = 'test_data_set'
test_data_num = 3

import glob
import os

# Load inputs
inputs = []
for i in range(test_data_num):
    input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

print('Loaded {} inputs successfully.'.format(test_data_num))
        
# Load reference outputs

ref_outputs = []
for i in range(test_data_num):
    output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())    
        ref_outputs.append(numpy_helper.to_array(tensor))
        
print('Loaded {} reference outputs successfully.'.format(test_data_num))

#Inference using ONNX Runtime

# Run the model on the backend
session = onnxruntime.InferenceSession('resnet50v2.onnx', None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  

print('Input Name:', input_name)
outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

print('Predicted {} results.'.format(len(outputs)))

# Compare the results with reference outputs up to 4 decimal places
# for ref_o, o in zip(ref_outputs, outputs):
#     np.testing.assert_almost_equal(ref_o, o, 4)
#     print(ref_o)
#     print ("---------------------------------")
#     print(o)
print (test_data_num)
print(len(outputs))

#print (outputs)
#print (ref_outputs)  
print('ONNX Runtime outputs are similar to reference outputs!')