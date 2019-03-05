#pragma once
#include "types.h"
#include "tensor_t.h"
#include <math.h>
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
struct quantization_params_16
{
	float scale;
	uint16_t zero_point;
	quantization_params_16(float _scale, uint16_t _zero_point)
	{
		this->scale = _scale;
		this->zero_point = _zero_point;
	}

	quantization_params_16(void)
	{
		this->scale = 0.f;
		this->zero_point = 0;
	}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct quantization_params_8
{
	float scale;
	uint16_t zero_point;
	quantization_params_8(float _scale, uint8_t _zero_point)
	{
		this->scale = _scale;
		this->zero_point = _zero_point;
	}

	quantization_params_8(void)
	{
		this->scale = 0.f;
		this->zero_point = 0;
	}
};
#pragma pack(pop)

void find_min_max(tensor_t<float> &input, float &min, float &max)
{
	min = max = input(0,0,0);
	for (int i = 0; i < input.size.x; i++)
	{
		for (int j = 0; j < input.size.y; j++)
		{
			for (int k = 0; k <input.size.z; k++)
			{
				const float val = input(i, j, k);
				min = std::min(min, val);
				max = std::max(max, val);
			}
		}
	}
}

void find_min_max(vector<tensor_t<float>> &input, float &min, float &max)
{
	min = max = input[0](0,0,0);
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
					min = std::min(min, val);
					max = std::max(max, val);
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

template <typename T, typename T1>
void choose_quantization_params(float min, float max, T &result, char bits = 8)
{
	min = std::min(min, 0.f);
	max = std::max(max, 0.f);

	assert (bits == 8 || bits == 16);
	const float qmin = 0;
	const float qmax = pow(2,bits) - 1;

	const double scale = (max - min) / (qmax - qmin);

	const double initial_zero_point = qmin - min / scale;

	T1 nudged_zero_point = 0;
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
		nudged_zero_point = static_cast<T1>(round(initial_zero_point));
	}

	result.scale = scale;
	result.zero_point = nudged_zero_point;
}

template <typename T, typename Tp>
void quantize(const Tp &qparams, const vector<float> &src,
			  vector<T> &dst, char bits = 8)
{
	assert (bits == 8 || bits == 16);
	const float qmax = (pow(2,bits)) -1 ;
	assert(src.size() == dst.size());
	for (size_t i = 0; i < src.size(); i++)
	{
		const float real_val = src[i];
		const float transformed_val = qparams.zero_point + real_val / qparams.scale;
		const float clamped_val = std::max(0.f, std::min(qmax, transformed_val));
		dst[i] = static_cast<T>(round(clamped_val));
	}
}

template <typename T, typename Tp>
void dequantize(const Tp &qparams,
				const vector<T> &src, vector<float> &dst)
{
	assert(src.size() == dst.size());
	for (size_t i = 0; i < src.size(); i++)
	{
		const T quantized_val = src[i];
		dst[i] = qparams.scale * (quantized_val - static_cast<T>(qparams.zero_point));
	}
}

void quantize_multiplier(float real_multiplier, int32_t &quantized_multiplier,
						int &right_shift)
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
	quantized_multiplier = static_cast<int32_t>(q);
	right_shift = s+32-1;
}