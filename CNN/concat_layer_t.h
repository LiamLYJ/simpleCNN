#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct concat_layer_t
{
	layer_type type = layer_type::concat;
	tensor_t<float> in_1;
	tensor_t<float> in_2;
	tensor_t<float> out;

	concat_layer_t(tdsize in_size_1, tdsize in_size_2)
		:
		in_1( in_size_1.x, in_size_1.y, in_size_1.z ),
		in_2( in_size_2.x, in_size_2.y, in_size_2.z ),
		out( in_size_1.x, in_size_1.y, in_size_1.z + in_size_2.z)
	{
		assert( in_size_1.x == in_size_2.x );
		assert( in_size_1.y == in_size_2.y );
	}

	void activate( tensor_t<float>& in_1, tensor_t<float>& in_2 )
	{
		this->in_1 = in_1;
		this->in_2 = in_2;
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
					if (z - in_1.size.z < 0) {
					    out( x, y, z ) = in_1(x, y, z);
					} else {
					    out( x, y, z ) = in_2(x, y, z - in_1.size.z);
					}
				}
			}
		}
	}

};
#pragma pack(pop)