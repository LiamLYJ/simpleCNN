#pragma once
#include "layer_t.h"

using namespace std;
#pragma pack(push, 1)
struct conv_layer_t
{
	layer_type type = layer_type::conv;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<uint8_t> out_fix;
	tensor_t<uint8_t> in_fix;
	quantization_params in_params;
	quantization_params out_params;
	quantization_params weight_params;
	quantization_params bias_params;
	vector<tensor_t<float>> filters;
	vector<tensor_t<uint8_t>> filters_fix;
	vector<float> bias;
	vector<uint8_t> bias_fix;
	uint16_t stride;
	uint16_t extend_filter;
	uint16_t padding;

	conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, uint16_t padding, tdsize in_size)
		:	in(in_size.x, in_size.y, in_size.z),
			in_fix(in_size.x, in_size.y, in_size.z),
		    out(
			  (in_size.x - extend_filter + 2 * padding) / stride + 1,
			  (in_size.y - extend_filter + 2 * padding) / stride + 1,
			  number_filters),
			out_fix(
			  (in_size.x - extend_filter + 2 * padding) / stride + 1,
			  (in_size.y - extend_filter + 2 * padding) / stride + 1,
			  number_filters),
			in_params(0.f, 0),
			out_params(0.f, 0),
			weight_params(0.f, 0),
			bias_params(0.f, 0)
	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		this->padding = padding;
		assert((float(in_size.x - extend_filter) / stride + 1) ==
			   ((in_size.x - extend_filter) / stride + 1));

		assert((float(in_size.y - extend_filter) / stride + 1) ==
			   ((in_size.y - extend_filter) / stride + 1));

		for (int a = 0; a < number_filters; a++)
		{
			tensor_t<float> t(extend_filter, extend_filter, in_size.z);
			tensor_t<uint8_t> t_fix(extend_filter, extend_filter, in_size.z);

			int maxval = extend_filter * extend_filter * in_size.z;

			for (int i = 0; i < extend_filter; i++)
				for (int j = 0; j < extend_filter; j++)
					for (int z = 0; z < in_size.z; z++)
                        {
                            t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
                            t_fix(i,j,z) = round(t(i,j,z));
                        }
 			filters.push_back(t);
            filters_fix.push_back(t_fix);
		}
	}

	point_t map_to_input(point_t out, int z)
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	
	tensor_t<float> conv_pad(tensor_t<float> in)
	{
		uint8_t x = in.size.x + 2 * padding;
		uint8_t y = in.size.y + 2 * padding;
		uint8_t z = in.size.z;

		if (padding > 0)
		{
			tensor_t<float> padded(x, y, z);
			padded.copy_from_padding(in);
			return padded;
		}
		else 
		{
			return in;
		}
	}

	template <typename T>
    void load_weights(vector<T> & data)
	{
		int c = this->filters.size();
		int z = this->filters[0].size.z;
		int y = this->filters[0].size.y;
		int x = this->filters[0].size.x;

        auto filter_p = this->filters.begin();
		auto pointer = data.begin();
		for (int v = 0; v < c; v++)
		{
			for (int i = 0; i < x; i++)
				for (int j = 0; j < y; j++)
					for (int k = 0; k < z; k++)
						{
						    (*filter_p)(i, j, k) = *pointer;
						    pointer ++;
						}
        filter_p ++;
		}
	}

	template <typename T>
    void load_weights(const vector<T> & data, vector<tensor_t<T>> & filters)
	{
		int c = filters.size();
		int z = filters[0].size.z;
		int y = filters[0].size.y;
		int x = filters[0].size.x;

        auto filter_p = filters.begin();
		auto pointer = data.begin();
		for (int v = 0; v < c; v++)
		{
			for (int i = 0; i < x; i++)
				for (int j = 0; j < y; j++)
					for (int k = 0; k < z; k++)
						{
						    (*filter_p)(i, j, k) = *pointer;
						    pointer ++;
						}
        filter_p ++;
		}
	}

    template <typename T>
    void from_weights(const vector<tensor_t<T>> & filters, vector<T> & data)
    {
        int c = filters.size();
        int z = filters[0].size.z;
        int y = filters[0].size.y;
        int x = filters[0].size.x;

        auto filter_p = filters.begin();
        auto pointer = data.begin();
        for (int v = 0; v < c; v++)
        {
            for (int i = 0; i < x; i++)
                for (int j = 0; j < y; j++)
                    for (int k = 0; k < z; k++)
                        {
                            *pointer = (*filter_p)(i,j,k);
                            pointer ++;
                        }
            filter_p ++;
        }
    }

	template <typename T>
    void load_bias(const vector<T> data)
	{
		this->bias.assign(data.begin(), data.end());
	}

	template <typename T>
    void load_bias(const vector<T> data, vector<T> bias)
	{
		bias.assign(data.begin(), data.end());
	}

	template <typename T>
    void from_bias(const vector<T> bias, vector<T> data)
	{
		data.assign(bias.begin(), bias.end());
	}


	void activate(tensor_t<float> &in)
	{
		this->in = conv_pad(in);
		activate();
	}

	void activate()
	{
		for (int filter = 0; filter < filters.size(); filter++)
		{
			tensor_t<float> &filter_data = filters[filter];
			for (int x = 0; x < out.size.x; x++)
			{
				for (int y = 0; y < out.size.y; y++)
				{
					point_t mapped = map_to_input({(uint16_t)x, (uint16_t)y, 0}, 0);
					float sum = 0;
					for (int i = 0; i < extend_filter; i++)
						for (int j = 0; j < extend_filter; j++)
							for (int z = 0; z < in.size.z; z++)
							{
								float f = filter_data(i, j, z);
								float v = in(mapped.x + i, mapped.y + j, z);
								sum += f * v;
							}
					out(x, y, filter) = sum + bias[filter];
				}
			}
		}
	}

	void fix_activate(tensor_t<float> &in)
	{
		this->in = conv_pad(in);
		activate();
		render_params();
		render_quantize();

		const float real_m = (this->in_params.scale * this->weight_params.scale) / this->out_params.scale;
		int32_t fake_m = 0;
		int right_shift = 0;
		quantize_multiplier(real_m, fake_m, right_shift);

		fix_multi_convolution(fake_m);
		// fix_add_bias();

	}

	void render_quantize()
	{
		vector<float> tmp_float;
		vector<uint8_t> tmp_uint8;
		// vector<float> _tmp_float;

        from_tensor(this->in, tmp_float);
        tmp_uint8.resize(tmp_float.size());
        quantize(this->in_params, tmp_float, tmp_uint8);
        // _tmp_float.resize(tmp_float.size());
        // dequantize(this->in_params, tmp_uint8, &_tmp_float);
        to_tensor(tmp_uint8, this->in_fix);

        this->from_weights(this->filters, tmp_float);
        tmp_uint8.resize(tmp_float.size());
        quantize(this->weight_params, tmp_float, tmp_uint8);
        this->load_weights(tmp_uint8, this->filters_fix);

        this->from_bias(this->bias, tmp_float);
        tmp_uint8.resize(tmp_float.size());
        quantize(this->bias_params, tmp_float, tmp_uint8);
        this->load_bias(tmp_uint8, this->bias_fix);

	}

	void render_params()
	{
		float value_min, value_max;
		find_min_max(this->in, value_min, value_max);
		this->in_params = choose_quantization_params(value_min, value_max);
		find_min_max(this->out, value_min, value_max);
		this->out_params = choose_quantization_params(value_min, value_max);
		find_min_max(this->filters, value_min, value_max);
		this->weight_params = choose_quantization_params(value_min, value_max);

        // as suggested in google's paper
		this->bias_params.zero_point = 0;
        this->bias_params.scale = this->in_params.scale * this->weight_params.scale;
	}

	void fix_multi_convolution(const int32_t &fake_m)
	{
		for (int filter = 0; filter < filters_fix.size(); filter++)
		{
			tensor_t<uint8_t> &filter_data = filters_fix[filter];
			for (int x = 0; x < out_fix.size.x; x++)
			{
				for (int y = 0; y < out_fix.size.y; y++)
				{
					point_t mapped = map_to_input({(uint16_t)x, (uint16_t)y, 0}, 0);
					float sum = 0;
					for (int i = 0; i < extend_filter; i++)
						for (int j = 0; j < extend_filter; j++)
							for (int z = 0; z < in_fix.size.z; z++)
							{
								float f = filter_data(i, j, z);
								float v = in_fix(mapped.x + i, mapped.y + j, z);
								sum += f * v;
							}
				}
			}
		}
	}

	void fix_add_bias()
	{

	}

};
#pragma pack(pop)
