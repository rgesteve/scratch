import onnx
import os
import sys
from quantize import quantize, QuantizationMode

input_onnx_model = sys.argv[1]
directory, model_name = os.path.split(input_onnx_model)

# Load the onnx model
model = onnx.load(input_onnx_model)

output_onnx_model = directory + "\\quantized_" + model_name

# Quantize
quantized_model = quantize(model, per_channel=False, quantization_mode=QuantizationMode.IntegerOps_Dynamic)
# Save the quantized model
onnx.save_model(quantized_model, output_onnx_model)