#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct upsample_layer_t
{
	layer_type type = layer_type::upsample;
	tensor_t<float> in;
	tensor_t<float> out;
	uint16_t stride;

	upsample_layer_t( uint16_t stride, tdsize in_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x * stride, in_size.y * stride, in_size.z )
	{
		this->stride = stride;
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int x = 0; x < out.size.x; x++ )
		{
			for ( int y = 0; y < out.size.y; y++ )
			{
				for ( int z = 0; z < out.size.z; z++ )
				{
					int in_x = x / stride;
					int in_y = y / stride;
					int in_z = z;
					float out_val = in(in_x, in_y, in_z);
					out( x, y, z ) = out_val;
				}
			}
		}
	}

};
#pragma pack(pop)