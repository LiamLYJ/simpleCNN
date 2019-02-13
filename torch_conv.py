import torch
import torch.nn as nn
import numpy as np
import re
import os
import torch.nn.functional as F

weight_fn = './data/weight_00.npy'
bias_fn = './data/bias_00.npy'
input_fn = './data/input.npy'

weight_np = np.load(weight_fn)
bias_np = np.load(bias_fn)
input_np = np.load(input_fn)

weight_shape =  weight_np.shape
bias_shape = bias_np.shape

out_c, in_c = weight_shape[0], weight_shape[1]
assert bias_shape[0] == out_c

conv_weight = torch.Tensor(weight_np)
conv_bias = torch.Tensor(bias_np)
conv_input = torch.Tensor(input_np)
conv_input = conv_input.unsqueeze(0).permute(0,3,1,2)

output = F.conv2d(input = conv_input, weight = conv_weight, bias = conv_bias, stride = 1, padding = 1) 
print (output.shape)

