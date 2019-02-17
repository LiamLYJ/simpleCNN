import torch
import torch.nn as nn
import numpy as np
import re
import os
import torch.nn.functional as F

# weight_fn = './data/weight_00.npy'
# bias_fn = './data/bias_00.npy'
# input_fn = './data/input.npy'

# weight_np = np.load(weight_fn)
# bias_np = np.load(bias_fn)
# input_np = np.load(input_fn)

# weight_shape =  weight_np.shape
# bias_shape = bias_np.shape

# out_c, in_c = weight_shape[0], weight_shape[1]
# assert bias_shape[0] == out_c

# conv_weight = torch.Tensor(weight_np)
# conv_bias = torch.Tensor(bias_np)
# conv_input = torch.Tensor(input_np)
# conv_input = conv_input.unsqueeze(0).permute(0,3,1,2)

# output = F.conv2d(input = conv_input, weight = conv_weight, bias = conv_bias, stride = 1, padding = 1) 
# print (output.shape)

# np.save('result_py.npy', output)



save_dir = 'test_data'
os.makedirs(save_dir, exist_ok=True)

input_data = np.array(
    [
        [2,0,0,1,0,
         1,1,2,1,1,
         0,1,0,0,0,
         2,0,2,2,2,
         0,1,0,1,2], 

        [1,2,2,2,1,
        0,0,1,1,0,
        2,2,1,0,2,
        0,1,0,1,2,
        1,2,0,1,1], 

        [0,2,0,0,1,
        2,1,0,0,0,
        0,1,0,2,2,
        0,1,2,0,0,
        2,2,1,1,0]
    ]
    )

input = input_data.reshape([1,3,5,5]) # torch shape : b,c,h,w
input_save = np.transpose(input.squeeze(), [1,2,0] ).astype(np.float32)
np.save(os.path.join(save_dir,'test_input.npy'), input_save)


# input_data = np.array(
#     [
#         [0,1,2,
#          3,4,5,
#          6,7,8,],

#         [9,10,11,
#         12,13,14,
#         15,16,17], 

#         [18,19,20,
#         21, 22, 23,
#         24,25,26,]
#     ]
#     )

# input = input_data.reshape([1,3,3,3]) # torch shape : b,c,h,w
# input_save = np.transpose(input.squeeze(), [2,1,0] ).astype(np.float32)
# np.save(os.path.join(save_dir,'test_input.npy'), input_save)


weight_data = np.array(
    [
        [[-1,1,1,
        -1,1,-1,
        0,1,0],

        [1,1,0,
        1,0,0,
        0,1,-1],

        [0,0,0,
        1,1,1,
        -1,0,-1]],


        [[1,-1,0,
        -1,-1,1,
        1,1,1],

        [-1,1,-1,
        0,1,1,
        -1,1,-1],

        [1,0,0,
        0,-1,-1,
        0,0,0]]
    ]
)
weight = weight_data.reshape([2,3,3,3]) # torch shape o_c,i_c, h,w
weight_save = np.transpose(weight, [0,2,3,1]).astype(np.float32)
np.save(os.path.join(save_dir, 'test_weight.npy'), weight_save)

bias_data = np.array([1,0])
bias_save = bias = bias_data.reshape(2).astype(np.float32)
np.save(os.path.join(save_dir, 'test_bias.npy'), bias_save)

conv_bias = torch.Tensor(bias) 
conv_weight = torch.Tensor(weight)
conv_input = torch.Tensor(input)
conv_output = F.conv2d(input = conv_input, weight = conv_weight, bias = conv_bias, stride = 2, padding = 1) 
# conv_output = F.conv2d(input = conv_input, weight = conv_weight, bias = conv_bias, stride = 1, padding = 0) 

output = conv_output.numpy().squeeze()
output_save = np.transpose(output, [1,2,0]).astype(np.float32)
np.save(os.path.join(save_dir, 'test_output.npy'), output_save)

