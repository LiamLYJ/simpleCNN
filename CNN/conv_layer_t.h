#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct conv_layer_t
{
	layer_type type = layer_type::conv;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<tensor_t<float>> filters;
	std::vector<float> bias;
	uint16_t stride;
	uint16_t extend_filter;
	uint16_t padding;

	conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, uint16_t padding, tdsize in_size)
		:	in(in_size.x, in_size.y, in_size.z),
		  out(
			  (in_size.x - extend_filter + 2 * padding) / stride + 1,
			  (in_size.y - extend_filter + 2 * padding) / stride + 1,
			  number_filters)
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

			int maxval = extend_filter * extend_filter * in_size.z;

			for (int i = 0; i < extend_filter; i++)
				for (int j = 0; j < extend_filter; j++)
					for (int z = 0; z < in_size.z; z++)
						t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
			filters.push_back(t);
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

    void load_weights(std::vector<float> data)
	{
		int c = this->filters.size();
		int z = this->filters[0].size.z;
		int y = this->filters[0].size.y;
		int x = this->filters[0].size.x;

    std::vector<tensor_t<float>>::iterator filter_p = this->filters.begin();
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

    void load_bias(std::vector<float> data)
	{
		this->bias.assign(data.begin(), data.end());
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

};
#pragma pack(pop)
