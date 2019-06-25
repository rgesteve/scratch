
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
import urllib.request
import time
import json
#Load sample inputs and outputs
input_filename = sys.argv[1]
output_filename = sys.argv[2]
print (input_filename + " " + output_filename)
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
        print (type(f))
        tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

#print('Loaded {} inputs successfully.'.format(test_data_num))
        
# Load reference outputs

ref_outputs = []
for i in range(test_data_num):
    output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())    
        ref_outputs.append(numpy_helper.to_array(tensor))
        
#print('Loaded {} reference outputs successfully.'.format(test_data_num))

#Inference using ONNX Runtime

# Run the model on the backend
session = onnxruntime.InferenceSession('model.onnx', None)



# get the name of the first input of the model
input_name = session.get_inputs()[0].name  
#print('Input Name:', input_name)

start = time.time()

outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

#end timer
end = time.time()

#print (outputs)
result = []
for i in outputs:
    result.append(int(np.argmax(np.array(i).squeeze(), axis=0)))

result_dict = {"result": result,
          "time_in_sec": end - start}

#print (result_dict)

ref_result = []
for i in ref_outputs:
    ref_result.append(int(np.argmax(np.array(i).squeeze(), axis=0)))

ref_outputs_dict = {"result": ref_result}
#print (ref_result)

#print('Predicted {} results.'.format(len(outputs)))


my_dict = { 'actual_output' : {}, 'ref_output' : {}}

count = 0
#Compare the results with reference outputs up to 4 decimal places
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o, 2)
    my_dict['actual_output'][count] = o.tolist()
    my_dict['actual_output'][count].append(int(np.argmax(np.array(o).squeeze(), axis=0)))
    my_dict['ref_output'][count] = ref_o.tolist()
    my_dict['ref_output'][count].append(int(np.argmax(np.array(ref_o).squeeze(), axis=0)))
    count +=1

with open ('C:\\output\\result.json', 'w') as json_file:
    json.dump(my_dict, json_file)
 
#print('ONNX Runtime outputs are similar to reference outputs!')