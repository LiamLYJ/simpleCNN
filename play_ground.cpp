
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

int test_save(void)
{
    const long unsigned leshape[] = {2, 3};
    vector<double> data{1, 2, 3, 4, 5, 6};
    npy::SaveArrayAsNumpy("data/out.npy", false, 2, leshape, data);

    const long unsigned leshape2[] = {6};
    npy::SaveArrayAsNumpy("data/out2.npy", false, 1, leshape2, data);

    return 0;
}

int main(int argc, char **argv)
{
    string fn_weight = "test_data/test_weight.npy";
    string fn_bias = "test_data/test_bias.npy";
    string fn_input = "test_data/test_input.npy";
    vector<float> weight_data;
    vector<float> bias_data;
    vector<float> input_data;
    vector<unsigned long> weight_shape;
    vector<unsigned long> bias_shape;
    vector<unsigned long> input_shape;

    LoadArrayFromNumpy(fn_weight, weight_shape, weight_data);
    LoadArrayFromNumpy(fn_bias, bias_shape, bias_data);
    LoadArrayFromNumpy(fn_input, input_shape, input_data);

    tensor_t<float> image_tensor(5, 5, 3);
    to_tensor(input_data, image_tensor);

    tdsize shape = {5,5,3};
    // conv_layer_t *layer = new conv_layer_t(1, 3, 2, 0, shape);
    conv_layer_t *layer = new conv_layer_t(2, 3, 2, 1, shape);
    layer->load_weights(weight_data);
    layer->load_bias(bias_data);

    layer->activate(image_tensor);
    tensor_t<float> out_tensor = layer->out;
    vector<float> out_data(3*3*2,0);
    const long unsigned leshape[] = {3, 3, 2};
    from_tensor(out_tensor, out_data);
    SaveArrayAsNumpy("test_data/output.npy", false, 3, leshape, out_data);
   
    return 0;
}


