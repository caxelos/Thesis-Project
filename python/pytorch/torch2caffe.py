## Check that tutorials here:
#https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
#a basic tutorial:https://github.com/BIGBALLON/Caffe2-Tutorial/tree/master/06_pytorch_to_caffe2
#https://www.learnopencv.com/pytorch-model-inference-using-onnx-and-caffe2/
# i use 2 inputs

##### PART 1: TORCH TO ONNX

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import importlib
module = importlib.import_module('models.resnet_preact')#models.{}'.format(args.arch))
model = module.Model()
model.load_state_dict(torch.load('model_state.pt'))
model.eval()
img = torch.rand(1,1,60,36)
pose = torch.rand(1,2)
traced_script_module = torch.jit.trace(model, (img,pose))#gia multiple input:https://towardsdatascience.com/model-summary-in-pytorch-b5a1e4b64d25
traced_script_module.save('model.pt')


# Export the model
torch_out = torch.onnx._export(model,             # model being run
                               (img,pose),                       # model input (or a tuple for multiple inputs)
                               "super_resolution.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

#1)convert an onnx model to caffe model’s files
#convert-onnx-to-caffe2 model.onnx --output predict_net.pb --init-net-output init_net.pb


##### PART 2: ONNX TO CAFFE2
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend#isws thelei:pip install future



# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("super_resolution.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2_backend.prepare(model)
# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: img.data.numpy(),
	 model.graph.input[1].name: pose.data.numpy()
}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")