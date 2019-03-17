#pragma once
#include "tensor_t.h"
#include "fc_layer.h"
#include "pool_layer_t.h"
#include "relu_layer_t.h"
#include "conv_layer_t.h"
#include "dropout_layer_t.h"
#include "upsample_layer_t.h"

static void activate( layer_t* layer, tensor_t<float>& in )
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->activate( in );
			return;
		case layer_type::relu:
			((relu_layer_t*)layer)->activate( in );
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->activate( in );
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->activate( in );
			return;
		case layer_type::dropout_layer:
			((dropout_layer_t*)layer)->activate( in );
			return;
		case layer_type::upsample:
		    ((upsample_layer_t*)layer)->activate( in );
		default:
			assert( false );
	}
}