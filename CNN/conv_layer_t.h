#pragma once
#include <numeric>
#include "layer_t.h"

// use for debugging 
#include <Eigen/Dense>
using namespace Eigen;

using namespace std;
#pragma pack(push, 1)
struct conv_layer_t
{
	layer_type type = layer_type::conv;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<uint8_t> out_fix;
	quantization_params in_params;
	quantization_params_16 out_params_without_bias;
	quantization_params_32 out_params_with_bias;
	quantization_params weight_params;
	quantization_params_32 bias_params;
	vector<tensor_t<float>> filters;
	vector<float> bias;
	uint16_t stride;
	uint16_t extend_filter;
	uint16_t padding;

	conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, uint16_t padding, tdsize in_size)
		:	in(in_size.x, in_size.y, in_size.z),
		    out(
			  (in_size.x - extend_filter + 2 * padding) / stride + 1,
			  (in_size.y - extend_filter + 2 * padding) / stride + 1,
			  number_filters),
			out_fix(
			  (in_size.x - extend_filter + 2 * padding) / stride + 1,
			  (in_size.y - extend_filter + 2 * padding) / stride + 1,
			  number_filters),
			in_params(0.f, 0),
			out_params_without_bias(0.f, 0),
			out_params_with_bias(0.f, 0),
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

			int maxval = extend_filter * extend_filter * in_size.z;

			for (int i = 0; i < extend_filter; i++)
				for (int j = 0; j < extend_filter; j++)
					for (int z = 0; z < in_size.z; z++)
                        {
                            t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
                        }
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

	void activate(char with_bias = 1)
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
					if (with_bias)
						out(x, y, filter) = sum + bias[filter];
					else
						out(x, y, filter) = sum;
				}
			}
		}
	}

	void add_bias2out()
	{
		for (int filter = 0; filter < filters.size(); filter++)
		{
			tensor_t<float> &filter_data = filters[filter];
			for (int x = 0; x < out.size.x; x++)
			{
				for (int y = 0; y < out.size.y; y++)
				{
					out(x, y, filter) += bias[filter];
				}
			}
		}
	}

	void fix_activate(tensor_t<float> &in)
	{
		render_params(in);

		// use for debugging
		// vector<uint16_t> out_fix_xxx(this->out.size.z * this->out.size.y * this->out.size.x);
		// vector<float> out_fix_yyy;
		// for (int z = 0; z <this->out.size.z; z++)
		// 	for (int y=0; y < this->out.size.y; y++)
		// 		for (int x= 0; x < this->out.size.x; x++)
		// 			out_fix_yyy.push_back(this->out(x,y,z));

		// quantize(this->out_params_without_bias, out_fix_yyy, out_fix_xxx, 16);
		// vector<float> out_fix_zzz(this->out.size.z * this->out.size.y * this->out.size.x);
		// dequantize(this->out_params_without_bias, out_fix_xxx, out_fix_zzz);


		int rows = this->filters.size();
		int depth = this->in.size.z * this->filters[0].size.y * this->filters[0].size.x;
		int cols = this->out.size.y * this->out.size.x;
		vector<vector <uint8_t>> w_left_mul_matrix(rows, vector<uint8_t>(depth, 1));
		vector<vector <uint8_t>> in_right_mul_matrix(depth, vector<uint8_t>(cols, 1));
		vector<uint32_t> bias_uint(rows, 1);
		render_quantize(w_left_mul_matrix, in_right_mul_matrix, bias_uint);

		// use for debugging
		// for (int i = 0; i < w_left_mul_matrix.size(); i++)
		// 	{
		// 		vector<float> _tmpx(w_left_mul_matrix[0].size(), 1);
		// 		dequantize(this->weight_params, w_left_mul_matrix[i], _tmpx);
		// 	}
		// for (int i = 0; i < in_right_mul_matrix.size(); i++)
		// 	{
		// 		vector<float> _tmpx(in_right_mul_matrix[0].size(), 1);
		// 		dequantize(this->in_params, in_right_mul_matrix[i], _tmpx);
		// 	}


		const float real_m = (this->in_params.scale * this->weight_params.scale) / this->out_params_without_bias.scale;
		int32_t fake_m = 0;
		int right_shift = 0ll;
		quantize_multiplier(real_m, fake_m, right_shift);

		const float real_m1 = this->out_params_without_bias.scale / this->out_params_with_bias.scale;
		int32_t fake_m1 = 0;
		fake_m1 = static_cast<int32_t>(lroundf(real_m1));

		const float real_m2 = this->bias_params.scale / this->out_params_with_bias.scale;
		int32_t fake_m2 = 0;
		fake_m2 = static_cast<int32_t>(lroundf(real_m2));

		vector<vector<uint32_t>> out_fix_vector(rows, vector<uint32_t>(cols,1));
		fix_multi_convolution(fake_m, right_shift, w_left_mul_matrix, in_right_mul_matrix, out_fix_vector);
		fix_add_bias(out_fix_vector, bias_uint, fake_m1, fake_m2);

		// rescale_to_8bit(); TODO


		vector<uint32_t> _out_fix;
		for (int i =0; i<out_fix_vector.size(); i++)
		{
			for (int j =0; j <out_fix_vector[0].size(); j++)
			{
				_out_fix.push_back(out_fix_vector[i][j]);
			}
		}

		vector<float> _tmp(out_fix_vector.size() * out_fix_vector[0].size(), 1);
		// dequantize(this->out_params_without_bias, _out_fix, _tmp);
		dequantize(this->out_params_with_bias, _out_fix, _tmp);
		for (auto pp : _tmp)
			{
				cout << pp << endl;
			}

		print_tensor(this->out);
	}

	template <typename T>
	void render_quantize(vector<vector<uint8_t>> &w_left_mul_matrix, 
						vector<vector<uint8_t>> &in_right_mul_matrix,
						vector<T> &bias_uint)
	{
		int rows = this->filters.size();
		int depth = this->in.size.z * this->filters[0].size.y * this->filters[0].size.x;
		int cols = this->out.size.y * this->out.size.x;
		vector<vector<float>> weight_matrix(rows, vector<float>(depth, 1));
		vector<vector<float>> in_matrix(depth, vector<float>(cols, 1));
		
		prepare_conv2mul(weight_matrix, in_matrix);

		for (int i = 0; i < weight_matrix.size(); i++)
			quantize(this->weight_params, weight_matrix[i], w_left_mul_matrix[i]);
		for (int i = 0; i < in_matrix.size(); i++)
			quantize(this->in_params, in_matrix[i], in_right_mul_matrix[i]);
		
		quantize(this->bias_params, this->bias, bias_uint);
	}

	void prepare_conv2mul(vector<vector<float>> &weight, vector<vector<float>> &in)
	{
		// refernec on im2col_cython in conv2mul.pyx	
		int C = this->in.size.z;
		int H = this->in.size.y;
		int W = this->in.size.x;
		int field_height = this->filters[0].size.y;
		int field_width = this->filters[0].size.x;
		int stride = this->stride;

		int HH = (H  - field_height) / stride + 1;
		int WW = (W  - field_width) / stride + 1;

		// weight: [rows, depth] , in [depth, cols]
		int rows = this->filters.size();
		int depth = C*field_height*field_width;
		int cols = HH*WW;

		// render weight_matrix
		for (int index=0; index< rows; index++)
		{
            weight[index].clear();
			for ( int k = 0; k < C; k++ )
				for ( int i = 0; i < field_width; i++ )
					for ( int j = 0; j < field_height; j++ )
						weight[index].push_back(this->filters[index](i,j,k));
		}

		// render in_matrix
		int _row;
		int _col;
		for (int yy = 0; yy < HH; yy++)
			for (int xx = 0; xx < WW; xx++)
				for (int jj = 0; jj < field_height; jj++)
					 for (int ii = 0; ii < field_width; ii++)
						for (int c =0; c < C; c++)
						{
							_row = c * field_width * field_height + ii * field_height + jj;
							_col = yy * WW + xx;
							in[_row][_col] = this->in(stride * xx + ii, stride * yy + jj, c);
						}
		
		// use for debugging
		// MatrixXf eigen_weight(rows, depth);
		// MatrixXf eigen_in(depth, cols);

		// for (int i =0; i < rows; i++)
		// {
		// 	for (int j =0; j < depth; j++)
		// 	{
		// 		eigen_weight(i,j) = weight[i][j];
		// 	}
		// }

		// for (int i =0; i < depth; i++)
		// {
		// 	for (int j=0; j <cols; j++)
		// 	{
		// 		eigen_in(i,j) = in[i][j];
		// 	}
		// }

		// MatrixXf eigen_result(rows, cols);
		// eigen_result = eigen_weight * eigen_in;
		// cout << "eigen_weight \n"<< eigen_weight << endl;
		// cout << "eigen_in \n"<< eigen_in << endl;
		// cout << "results is \n" << eigen_result<< endl;
		// cout << "original is\n" <<endl;
		// print_tensor(this->out);
	}

	void render_params(tensor_t<float> &in)
	{
		this->in = conv_pad(in);

		float value_min, value_max;
		find_min_max(this->in, value_min, value_max);
		choose_quantization_params<quantization_params, uint8_t>(value_min, value_max, this->in_params);

		char with_bias = 0;
		// compute float activate to get output_params, do it offline
		activate(with_bias);
		// print_tensor(this->out);
		find_min_max(this->out, value_min, value_max);
		choose_quantization_params<quantization_params_16, uint16_t>(value_min, value_max, this->out_params_without_bias);
		add_bias2out();
		// print_tensor(this->out);
		find_min_max(this->out, value_min, value_max);
		choose_quantization_params<quantization_params_32, uint32_t>(value_min, value_max, this->out_params_with_bias);

		find_min_max(this->filters, value_min, value_max);
		choose_quantization_params<quantization_params, uint8_t>(value_min, value_max, this->weight_params);

		// bias.zero_point bias.scale is compute as fixed one as in google's paper
		this->bias_params.zero_point = 0;
        this->bias_params.scale = this->in_params.scale * this->weight_params.scale;

		// find_min_max(this->bias, value_min, value_max);
		// choose_quantization_params<quantization_params_32, uint32_t>(value_min, value_max, this->bias_params);
	}

	void fix_multi_convolution(const int32_t &fake_m, const int &right_shift,  
							const vector<vector<uint8_t>> &left_w, 
							const vector<vector<uint8_t>> &right_in,
							vector<vector<uint32_t>> &result_out)
	{
		int rows = left_w.size();
		int depth = left_w[0].size();
		assert (depth == right_in.size());
		int cols = right_in[0].size();
		assert ((rows>0) && (depth> 0) && (cols >0));

		const float M = static_cast<float>(fake_m) / (1ll << right_shift);
		for (int i =0; i< rows; i++)
		{
			for (int k =0; k < cols; k++)
			{
				//google method1 
				// int tmp_sum = 0;
				// for (int j=0; j<depth; j++)
				// {
				// 	tmp_sum += (left_w[i][j] - this->weight_params.zero_point) * (right_in[j][k] - this->in_params.zero_point);
				// }
				// result_out[i][k] = static_cast<uint32_t>(tmp_sum * M) + this->out_params_without_bias.zero_point;


				// google method 2
				int tmp_sum = 0;
				for (int j =0; j < depth; j++)
				{
					tmp_sum += left_w[i][j] * right_in[j][k];
				}
				int a_1 = accumulate(left_w[i].begin(), left_w[i].end(), 0);
				int a_2 = 0;	
				for (int tmp_index = 0; tmp_index < depth; tmp_index++) 
				{
					a_2 += right_in[tmp_index][k];
				}
	
				int _tmp = depth * this->weight_params.zero_point * this->in_params.zero_point -
							this->weight_params.zero_point * a_2 - this->in_params.zero_point * a_1 + tmp_sum;
				result_out[i][k] = static_cast<uint32_t>(_tmp * M) + this->out_params_without_bias.zero_point;

			}
		}
	}

	template <typename T>
	void fix_add_bias(vector<vector<uint32_t>> &result_out, vector<T> bias_uint, 
						const int32_t &fake_m1, const int32_t &fake_m2) 
	{
		const float M1 = static_cast<float> (fake_m1);
		const float M2 = static_cast<float> (fake_m2);

		int rows = result_out.size();
		int cols = result_out[0].size();
		for (int i =0; i <rows; i++)
			for (int j =0; j <cols; j++)
			{
				result_out[i][j] = M1 * (static_cast<long>(result_out[i][j]) - this->out_params_without_bias.zero_point) 
						+ M2 * (static_cast<long>(bias_uint[i]) - this->bias_params.zero_point)
						+ this->out_params_with_bias.zero_point;
			}
	}

	void rescale_to_8bit()
	{

	}

};
#pragma pack(pop)
