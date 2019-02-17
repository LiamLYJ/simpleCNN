#pragma once
#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct tensor_t
{
	T * data;

	tdsize size;

	tensor_t( int _x, int _y, int _z )
	{
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	tensor_t( const tensor_t& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
		);
		this->size = other.size;
	}

	tensor_t<T>& operator=( const tensor_t<T>& other )
	{

		if (this == &other)
			return *this;
		delete data; 
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
		);
		this->size = other.size;
		return *this;
	}

	tensor_t<T> operator+( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z )
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	void copy_from( std::vector<std::vector<std::vector<T>>> data )
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for ( int i = 0; i < x; i++ )
			for ( int j = 0; j < y; j++ )
				for ( int k = 0; k < z; k++ )
					get( i, j, k ) = data[k][j][i];
	}

    void copy_from_padding( tensor_t<float> data, uint8_t padding)
    {
        int x = data.size.x;
        int y = data.size.y;
        int z = data.size.z;
        
        for ( int i = padding; i < x - padding; i++ )
            for ( int j = padding; j < y - padding; j++ )
                for ( int k = 0; k < z; k++ )
                    get( i, j, k ) = data(i,j,k);

        for ( int i = 0; i < padding; i++ )
            for ( int j = 0; j < padding; j++ )
                for ( int k = 0; k < z; k++ )
                    get( i, j, k ) = 0;
        for ( int i = x - padding; i < x; i++ )
            for ( int j = y - padding; j < y; j++ )
                for ( int k = 0; k < z; k++ )
                    get( i, j, k ) = 0;
    }
    
	~tensor_t()
	{
		delete[] data;
	}
};

static void print_tensor( tensor_t<float>& data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
		printf( "[Dim%d]\n", z );
		for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
				printf( "%.2f \t", (float)data.get( x, y, z ) );
			}
			printf( "\n" );
		}
	}
}

static void to_tensor( std::vector<float> data, tensor_t<float> & tensor)
{
	int x = tensor.size.x;
	int y = tensor.size.y;
	int z = tensor.size.z;

	auto pointer = data.begin();
	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				{
					tensor( i, j, k ) = *pointer; 
					pointer ++;
				}
}

static void from_tensor( tensor_t<float> tensor, std::vector<float> & data)
{
	int x = tensor.size.x;
	int y = tensor.size.y;
	int z = tensor.size.z;

	auto pointer = data.begin();
	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				{
				*pointer = tensor(i,j,k);
				pointer ++;
				}
}
