import torch
import numpy as np
import re
import os
from scipy.misc import imread, imresize


def shrink_weight_bias(conv_weight_key,model, bn_eps = 1e-6):
    index = re.findall('module_list.(\d+).', conv_weight_key)[0]
    shape = model[conv_weight_key].shape
    out_channels, in_channels = shape[0], shape[1]
   
    w_conv = model[conv_weight_key]
    conv_bias_key = 'module_list.%s.conv_%s.bias'%(index, index)
    try:
        b_conv = model(conv_bias_key)
    except:
        b_conv = torch.zeros(out_channels)

    bn_weight_key = 'module_list.%s.batch_norm_%s.weight'%(index,index)
    bn_bias_key = 'module_list.%s.batch_norm_%s.bias'%(index, index)
    bn_running_mean_key = 'module_list.%s.batch_norm_%s.running_mean'%(index, index)
    bn_running_var_key = 'module_list.%s.batch_norm_%s.running_var'%(index, index)

    if bn_weight_key in model.keys():
        w_bn = model[bn_weight_key]
        b_bn = model[bn_bias_key]
        running_mean = model[bn_running_mean_key]
        running_var = model[bn_running_var_key]

        w_conv = w_conv.view(out_channels, -1)
        _w_bn = torch.diag(w_bn.div(torch.sqrt(bn_eps+running_var)))
        fused_weight = torch.mm(_w_bn, w_conv).view(shape)

        b_bn = b_bn - w_bn.mul(running_mean).div(torch.sqrt(running_var + bn_eps))
        fused_bias = b_conv + b_bn        
    else:
        fused_weight = w_conv
        fused_bias = b_conv

    return fused_weight, fused_bias, shape


model_name = 'model_608.ckpt'
save_dir = '../data'
os.makedirs(save_dir, exist_ok=True)
fn = os.path.join(save_dir, 'model.npy')
bn_eps = 1e-6

model = torch.load(model_name, map_location = lambda storage, loc: storage)
model_keylist = list(model.keys())
model_keylist = sorted(model_keylist, key = lambda x: int(re.findall('module_list.(\d+).',x)[0]))

my_dict = {}
count = 0
for key in model_keylist:
    if ('conv' in key) and ('weight' in key):
        weight, bias, shape = shrink_weight_bias(key, model, bn_eps)
        my_dict['weight_%02d'%(count)] = weight.numpy()
        my_dict['bias_%02d'%(count)] = bias.numpy()
        np.save(os.path.join(save_dir,'weight_%02d.npy'%(count)), weight)
        np.save(os.path.join(save_dir,'bias_%02d.npy'%(count)), bias)
        count += 1
np.save(fn, my_dict)

test_jpg = imread('img.jpg')
test_jpg = imresize(test_jpg, (224,224)).astype(np.float32)
np.save(os.path.join(save_dir, 'input.npy'), test_jpg)
