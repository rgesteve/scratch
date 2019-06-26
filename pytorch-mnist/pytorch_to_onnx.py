from torch.autograd import Variable
import torch.onnx
import torchvision

input_pytorch_model = 'test.pt'
dummy_model_input = Variable(torch.randn(64, 784))

output_onnx_model = 'pytorch_mnist_model.onnx'

# load the PyTorch model
model = torch.load(input_pytorch_model)

# export the PyTorch model as an ONNX protobuf
torch.onnx.export(model, dummy_model_input, output_onnx_model)