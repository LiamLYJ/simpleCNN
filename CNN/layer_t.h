#pragma once
#include "types.h"
#include "tensor_t.h"
using namespace std;

#pragma pack(push, 1)
struct layer_t
{
	layer_type type;
	tensor_t<float> in;
	tensor_t<float> out;
};
#pragma pack(pop)


#pragma pack(push, 1)
struct quantization_params
{
	float scale;
	uint8_t zero_point;
	quantization_params(float _scale, uint8_t _zero_point)
	{
		this->scale = _scale;
		this->zero_point = _zero_point;
	}

	quantization_params(void)
	{
		this->scale = 0.f;
		this->zero_point = 0;
	}
};
#pragma pack(pop)


void find_min_max(tensor_t<float> &input, float* min, float* max)
{
	*min = *max = input(0,0,0);
	for (int i = 0; i < input.size.x; i++)
	{
		for (int j = 0; j < input.size.y; j++)
		{
			for (int k = 0; k <input.size.z; k++)
			{
				const float val = input(i, j, k);
				*min = std::min(*min, val);
				*max = std::max(*max, val);
			}
		}
	}
}

void find_min_max(vector<tensor_t<float>> &input, float *min, float *max)
{
	*min = *max = input[0](0,0,0);
	for (auto it = input.begin(); it != input.end(); it++)
	{
		tensor_t<float> item = *it;
		for (int i = 0; i < item.size.x; i++)
		{
			for (int j = 0; j < item.size.y; j++)
			{
				for (int k = 0; k <item.size.z; k++)
				{
					const float val = item(i, j, k);
					*min = std::min(*min, val);
					*max = std::max(*max, val);
				}
			}
		}
	}
}

void find_min_max(const vector<float> &input, float *min, float *max)
{
	*min = *max = input[0];
	for (auto it = input.begin(); it != input.end(); it++)
	{
		const float val = *it;
		*min = std::min(*min, val);
		*max = std::max(*max, val);
	}
}

quantization_params choose_quantization_params(float min, float max)
{
	min = std::min(min, 0.f);
	max = std::max(max, 0.f);

	const float qmin = 0;
	const float qmax = 255;

	const double scale = (max - min) / (qmax - qmin);

	const double initial_zero_point = qmin - min / scale;

	uint8_t nudged_zero_point = 0;
	if (initial_zero_point < qmin)
	{
		nudged_zero_point = qmin;
	}
	else if (initial_zero_point > qmax)
	{
		nudged_zero_point = qmax;
	}
	else
	{
		nudged_zero_point = static_cast<uint8_t>(round(initial_zero_point));
	}

	quantization_params result(0.f, 0);
	result.scale = scale;
	result.zero_point = nudged_zero_point;
	return result;
}


void quantize(const quantization_params &qparams, const vector<float> &src,
			  vector<uint8_t> *dst)
{
	assert(src.size() == dst->size());
	for (size_t i = 0; i < src.size(); i++)
	{
		const float real_val = src[i];
		const float transformed_val = qparams.zero_point + real_val / qparams.scale;
		const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
		(*dst)[i] = static_cast<uint8_t>(round(clamped_val));
	}
}

void dequantize(const quantization_params &qparams,
				const vector<uint8_t> &src, vector<float> *dst)
{
	assert(src.size() == dst->size());
	for (size_t i = 0; i < src.size(); i++)
	{
		const uint8_t quantized_val = src[i];
		(*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);
	}
}

void quantize_multiplier(float real_multiplier, int32_t *quantized_multiplier,
						int *right_shift)
{
	assert(real_multiplier > 0.f);
	assert(real_multiplier < 1.f);
	int s = 0;
	while (real_multiplier < 0.5f)
	{
		real_multiplier *= 2.0f;
		s++;
	}
	int64_t q = static_cast<int64_t>(round(real_multiplier * (1ll << 31)));
	assert(q <= (1ll << 31));
	if (q == (1ll << 31))
	{
		q /= 2;
		s--;
	}
	assert(s >= 0);
	assert(q <= std::numeric_limits<int32_t>::max());
	*quantized_multiplier = static_cast<int32_t>(q);
	*right_shift = s;
}