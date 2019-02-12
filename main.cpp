//
//  main.cpp
//  SimpleCNN
//
//  Created by lyj on 2019/02/07.
//  Copyright © 2019 lyj. All rights reserved.
//

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "CNN/cnn.h"
#include "npy.h"

using namespace std;
using namespace npy;

template<typename T>
void load_npy(string fn, vector <T> & data)
{
    vector<unsigned long> shape;

    npy::LoadArrayFromNumpy(fn, shape, data);

    cout << "layer:" << fn << endl;
    cout << "shape: ";
    for (size_t i = 0; i < shape.size(); i++)
        cout << shape[i] << ", ";
    cout << endl;
}


int main()
{
    vector<float> input_data;
    vector<float> float_data;
    vector<unsigned char> char_data;

    vector<layer_t *> layers;
    tdsize data_input = {224, 224, 3}; 
    conv_layer_t *layer_0 = new conv_layer_t(1, 3, 16, 1, data_input); // 224 * 224 * 3 -> 224 * 224 * 16
    relu_layer_t *layer_0_relu = new relu_layer_t(layer_0->out.size);
    pool_layer_t *layer_1 = new pool_layer_t(2, 2, layer_0_relu->out.size); // 224 * 224 * 16 -> 112 * 112 * 16
    string layer_weights_fn = "./data/weight_00.npy";
    load_npy(layer_weights_fn, float_data);
    layer_0->load_weights(float_data);
    string layer_bias_fn = "./data/bias_00.npy";
    load_npy(layer_bias_fn, float_data);
    layer_0->load_bias(float_data);
    
    conv_layer_t *layer_2 = new conv_layer_t(1, 3, 32, 1,layer_1->out.size); // 112 * 112 * 16 -> 112 * 112 * 32
    relu_layer_t *layer_2_relu = new relu_layer_t(layer_2->out.size);
    pool_layer_t *layer_3 = new pool_layer_t(2, 2, layer_2_relu->out.size); // 112 * 112 * 32 -> 64 * 64 * 32
    layer_weights_fn = "./data/weight_01.npy";
    load_npy(layer_weights_fn, float_data);
    layer_2->load_weights(float_data);
    layer_bias_fn = "./data/bias_01.npy";
    load_npy(layer_bias_fn, float_data);
    layer_2->load_bias(float_data);

    conv_layer_t *layer_4 = new conv_layer_t(1, 3, 64, 1,layer_3->out.size); // 64 * 64 * 32 -> 64 * 64 * 64
    relu_layer_t *layer_4_relu = new relu_layer_t(layer_4->out.size);
    pool_layer_t *layer_5 = new pool_layer_t(2, 2, layer_4_relu->out.size); // 64 * 64 * 64 -> 32 * 32 * 64
    layer_weights_fn = "./data/weight_02.npy";
    load_npy(layer_weights_fn, float_data);
    layer_4->load_weights(float_data);
    layer_bias_fn = "./data/bias_02.npy";
    load_npy(layer_bias_fn, float_data);
    layer_4->load_bias(float_data);

    conv_layer_t *layer_6 = new conv_layer_t(1, 3, 128,1, layer_5->out.size); // 32 * 32 * 64 -> 32 * 32 * 128 
    relu_layer_t *layer_6_relu = new relu_layer_t(layer_6->out.size);
    pool_layer_t *layer_7 = new pool_layer_t(2, 2, layer_6_relu->out.size); // 32 * 32 * 128 -> 16 * 16 * 128
    layer_weights_fn = "./data/weight_03.npy";
    load_npy(layer_weights_fn, float_data);
    layer_6->load_weights(float_data);
    layer_bias_fn = "./data/bias_03.npy";
    load_npy(layer_bias_fn, float_data);
    layer_6->load_bias(float_data);

    conv_layer_t *layer_8 = new conv_layer_t(1, 3, 128,1, layer_7->out.size); // 16 * 16 * 128 -> 16 * 16 * 128 
    relu_layer_t *layer_8_relu = new relu_layer_t(layer_8->out.size);
    pool_layer_t *layer_9 = new pool_layer_t(2, 2, layer_8_relu->out.size); // 16 * 16 * 128 -> 8 * 8 * 128
    layer_weights_fn = "./data/weight_04.npy";
    load_npy(layer_weights_fn, float_data);
    layer_8->load_weights(float_data);
    layer_bias_fn = "./data/bias_04.npy";
    load_npy(layer_bias_fn, float_data);
    layer_8->load_bias(float_data);

    conv_layer_t *layer_10 = new conv_layer_t(1, 3, 128,1, layer_9->out.size); // 8 * 8 * 128 -> 8 * 8 * 256
    relu_layer_t *layer_10_relu = new relu_layer_t(layer_10->out.size);
    layer_weights_fn = "./data/weight_05.npy";
    load_npy(layer_weights_fn, float_data);
    layer_10->load_weights(float_data);
    layer_bias_fn = "./data/bias_05.npy";
    load_npy(layer_bias_fn, float_data);
    layer_10->load_bias(float_data);   

    conv_layer_t *layer_11 = new conv_layer_t(1, 3, 128,1, layer_10->out.size); // 8 * 8 * 128 -> 8 * 8 * 24
    layer_weights_fn = "./data/weight_06.npy";
    load_npy(layer_weights_fn, float_data);
    layer_11->load_weights(float_data);
    layer_bias_fn = "./data/bias_06.npy";
    load_npy(layer_bias_fn, float_data);
    layer_11->load_bias(float_data);

    layers.push_back((layer_t *)layer_0);
    layers.push_back((layer_t *)layer_0_relu);
    layers.push_back((layer_t *)layer_1);
    layers.push_back((layer_t *)layer_2);
    layers.push_back((layer_t *)layer_2_relu);
    layers.push_back((layer_t *)layer_3);
    layers.push_back((layer_t *)layer_4);
    layers.push_back((layer_t *)layer_4_relu);
    layers.push_back((layer_t *)layer_5);
    layers.push_back((layer_t *)layer_6);
    layers.push_back((layer_t *)layer_6_relu);
    layers.push_back((layer_t *)layer_7);
    layers.push_back((layer_t *)layer_8);
    layers.push_back((layer_t *)layer_8_relu);
    layers.push_back((layer_t *)layer_9);
    layers.push_back((layer_t *)layer_10);
    layers.push_back((layer_t *)layer_10_relu);
    layers.push_back((layer_t *)layer_11);

    string input_fn = "./data/input.npy";
    load_npy(input_fn, char_data);
    input_data.assign(char_data.begin(), char_data.end());


    return 0;
}
