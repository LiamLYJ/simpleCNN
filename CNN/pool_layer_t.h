#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct pool_layer_t
{
	layer_type type = layer_type::pool;
	tensor_t<float> in;
	tensor_t<float> out;
	uint16_t stride;
	uint16_t extend_filter;

	pool_layer_t( uint16_t stride, uint16_t extend_filter, tdsize in_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out(
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			in_size.z
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );
	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

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
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float mval = -FLT_MAX;
					for ( int i = 0; i < extend_filter; i++ )
						for ( int j = 0; j < extend_filter; j++ )
						{
							float v = in( mapped.x + i, mapped.y + j, z );
							if ( v > mval )
								mval = v;
						}
					out( x, y, z ) = mval;
				}
			}
		}
	}

};
#pragma pack(pop)