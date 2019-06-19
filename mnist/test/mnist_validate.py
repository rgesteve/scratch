
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
import urllib.request
import time
import json
import sys
import os

from os.path import isfile, join
import glob



#Load sample inputs and outputs
input_dir = sys.argv[1]
output_dir = sys.argv[2]
print (input_dir + " " + output_dir)
# Load inputs


input_files = [join(input_dir,f) for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
output_files = [join(output_dir,f)  for f in os.listdir(output_dir) if isfile(join(output_dir, f))]


inputs = []
for input_file in input_files:
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        print (type(f))
        tensor.ParseFromString(f.read())
        #print(numpy_helper.to_array(tensor))
        test = numpy_helper.to_array(tensor)
        print (test.shape)
        #print (test[0,0,:,:])
        #print(type(numpy_helper.to_array(tensor)))
        print ("-----------------------------")
        print (test.dtype)
        #print(tensor)
        inputs.append(numpy_helper.to_array(tensor))

#print('Loaded {} inputs successfully.'.format(test_data_num))
        
# Load reference outputs
print(inputs)

ref_outputs = []
for output_file in output_files:
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())    
        test = numpy_helper.to_array(tensor)
        print (test.shape)
        print (test)
        ref_outputs.append(numpy_helper.to_array(tensor))

#Inference using ONNX Runtime

# Run the model on the backend
session = onnxruntime.InferenceSession('model.onnx', None)



# get the name of the first input of the model
input_name = session.get_inputs()[0].name  


start = time.time()

outputs = [session.run([], {input_name: i})[0] for i in inputs]

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