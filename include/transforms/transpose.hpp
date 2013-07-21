/*
 *  Copyright 2012 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/* 
 * This file contains a CUDA implementation of the array transpose operation.
 * 
 * Parts of this file are based on the transpose implementation in the
 * NVIDIA CUDA SDK.
 */

#pragma once

namespace cuda_specs {
	enum { MAX_GRID_DIMENSION = 65535 };
}

template<typename T>
struct Transpose {
	enum {
//#if __CUDA_ARCH__ < 200
		TILE_DIM     = 32,
//#else
//		TILE_DIM     = 64,
//#endif
		BLOCK_ROWS   = 8//,
		//PAD_MULTIPLE = TILE_DIM
	};
	
	Transpose() {}
	
	void transpose(const T* in,
				   size_t width, size_t height,
				   size_t in_stride, size_t out_stride,
				   T* out,
				   cudaStream_t stream=0);
	void transpose(const T* in,
				   size_t width, size_t height,
				   T* out,
				   cudaStream_t stream=0) {
		transpose(in, width, height, width, height, out, stream);
	}
private:
	// TODO: These should probably be imported from somewhere else
	template<typename U>
	inline U min(const U& a, const U& b) {
		return a < b ? a : b;
	}
	template<typename U>
	inline U round_up_pow2(const U& a) {
		U r = a-1;
		for( unsigned long i=1; i<=sizeof(U)*8/2; i<<=1 ) r |= r >> i;
		return r+1;
	}
	template<typename U>
	inline U round_down_pow2(const U& a) {
		return round_up_pow2(a+1)/2;
	}
	inline unsigned int log2(unsigned int a) {
		unsigned int r;
		unsigned int shift;
		r =     (a > 0xFFFF) << 4; a >>= r;
		shift = (a > 0xFF  ) << 3; a >>= shift; r |= shift;
		shift = (a > 0xF   ) << 2; a >>= shift; r |= shift;
		shift = (a > 0x3   ) << 1; a >>= shift; r |= shift;
		r |= (a >> 1);
		return r;
	}
	inline unsigned long log2(unsigned long a) {
		unsigned long r;
		unsigned long shift;
		r =     (a > 0xFFFFFFFF) << 5; a >>= r;
		shift = (a > 0xFFFF    ) << 4; a >>= shift; r |= shift;
		shift = (a > 0xFF      ) << 3; a >>= shift; r |= shift;
		shift = (a > 0xF       ) << 2; a >>= shift; r |= shift;
		shift = (a > 0x3       ) << 1; a >>= shift; r |= shift;
		r |= (a >> 1);
		return r;
	}
};

typedef unsigned int gpu_size_t;

template<bool GRID_IS_POW2, typename T>
__global__
void transpose_kernel(const T* in,
		      gpu_size_t width, gpu_size_t height,
		      gpu_size_t in_stride, gpu_size_t out_stride,
		      T* out,
		      gpu_size_t block_count_x,
		      gpu_size_t block_count_y,
		      gpu_size_t log2_gridDim_y)
{
	__shared__ T tile[Transpose<T>::TILE_DIM][Transpose<T>::TILE_DIM+1];
	
	gpu_size_t blockIdx_x, blockIdx_y;
	
	// Do diagonal index reordering to avoid partition camping in device memory
	if( width == height ) {
		blockIdx_y = blockIdx.x;
		if( !GRID_IS_POW2 ) {
			blockIdx_x = (blockIdx.x+blockIdx.y) % gridDim.x;
		}
		else {
			blockIdx_x = (blockIdx.x+blockIdx.y) & (gridDim.x-1);
		}
	}
	else {
		gpu_size_t bid = blockIdx.x + gridDim.x*blockIdx.y;
		if( !GRID_IS_POW2 ) {
			blockIdx_y = bid % gridDim.y;
			blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
		}
		else {
			blockIdx_y = bid & (gridDim.y-1);
			blockIdx_x = ((bid >> log2_gridDim_y) + blockIdx_y) & (gridDim.x-1);
		}
	}
	
	// Cull excess blocks (there may be many if we round up to a power of 2)
	if( blockIdx_x >= block_count_x ||
		blockIdx_y >= block_count_y ) {
		return;
	}
	
	gpu_size_t index_in_x = blockIdx_x * Transpose<T>::TILE_DIM + threadIdx.x;
	gpu_size_t index_in_y = blockIdx_y * Transpose<T>::TILE_DIM + threadIdx.y;
	gpu_size_t index_in = index_in_x + (index_in_y)*in_stride;
	
#pragma unroll
	for( gpu_size_t i=0; i<Transpose<T>::TILE_DIM; i+=Transpose<T>::BLOCK_ROWS ) {
		// TODO: Is it possible to cull some excess threads early?
		//if( index_in_x < width && index_in_y+i < height )
		tile[threadIdx.y+i][threadIdx.x] = in[index_in+i*in_stride];
	}
	
	__syncthreads();
	
	gpu_size_t index_out_x = blockIdx_y * Transpose<T>::TILE_DIM + threadIdx.x;
	// Avoid excess threads
	if( index_out_x >= height ) return;
	gpu_size_t index_out_y = blockIdx_x * Transpose<T>::TILE_DIM + threadIdx.y;
	gpu_size_t index_out = index_out_x + (index_out_y)*out_stride;
	
#pragma unroll
	for( gpu_size_t i=0; i<Transpose<T>::TILE_DIM; i+=Transpose<T>::BLOCK_ROWS ) {
		// Avoid excess threads
		if( index_out_y+i < width ) {
			out[index_out+i*out_stride] = tile[threadIdx.x][threadIdx.y+i];
		}
	}
}

template<typename T>
void Transpose<T>::transpose(const T* in,
			     size_t width, size_t height,
			     size_t in_stride, size_t out_stride,
			     T* out,
			     cudaStream_t stream)
{
	// Parameter checks
	// TODO: Implement some sort of error returning!
	if( 0 == width || 0 == height ) return;
	if( 0 == in ) return; //throw std::runtime_error("Transpose: in is NULL");
	if( 0 == out ) return; //throw std::runtime_error("Transpose: out is NULL");
	if( width > in_stride )
		return; //throw std::runtime_error("Transpose: width exceeds in_stride");
	if( height > out_stride )
		return; //throw std::runtime_error("Transpose: height exceeds out_stride");
	
	// Specify thread decomposition (uses up-rounded divisions)
	dim3 tot_block_count((width-1)  / TILE_DIM + 1,
						 (height-1) / TILE_DIM + 1);
	
	size_t max_grid_dim = round_down_pow2((size_t)cuda_specs::MAX_GRID_DIMENSION);
	
	// Partition the grid into chunks that the GPU can accept at once
	for( size_t block_y_offset = 0;
		 block_y_offset < tot_block_count.y;
		 block_y_offset += max_grid_dim ) {
		
		dim3 block_count;
		
		// Handle the possibly incomplete final grid
		block_count.y = min(max_grid_dim,
							tot_block_count.y - block_y_offset);
		
		for( size_t block_x_offset = 0;
			 block_x_offset < tot_block_count.x;
			 block_x_offset += max_grid_dim ) {
			
			// Handle the possibly incomplete final grid
			block_count.x = min(max_grid_dim,
								tot_block_count.x - block_x_offset);
			
			// Compute the chunked parameters
			size_t x_offset = block_x_offset * TILE_DIM;
			size_t y_offset = block_y_offset * TILE_DIM;
			size_t in_offset = x_offset + y_offset*in_stride;
			size_t out_offset = y_offset + x_offset*out_stride;
			size_t w = min(max_grid_dim*TILE_DIM, width-x_offset);
			size_t h = min(max_grid_dim*TILE_DIM, height-y_offset);
			
			dim3 block(TILE_DIM, BLOCK_ROWS);
			
			// TODO: Unfortunately there are cases where rounding to a power of 2 becomes
			//       detrimental to performance. Could work out a heuristic.
			//bool round_grid_to_pow2 = false;
			bool round_grid_to_pow2 = true;
			
			// Dispatch on grid-rounding
			if( round_grid_to_pow2 ) {
				dim3 grid(round_up_pow2(block_count.x),
						  round_up_pow2(block_count.y));
				// Run the CUDA kernel
				transpose_kernel<true><<<grid, block, 0, stream>>>
					(in + in_offset,
					 w, h,
					 in_stride, out_stride,
					 out + out_offset,
					 block_count.x,
					 block_count.y,
					 log2(grid.y));
			}
			else {
				dim3 grid(block_count.x, block_count.y);
				// Run the CUDA kernel
				transpose_kernel<false><<<grid, block, 0, stream>>>
					(in + in_offset,
					 w, h,
					 in_stride, out_stride,
					 out + out_offset,
					 block_count.x,
					 block_count.y,
					 log2(grid.y));
			}
			
#ifndef NDEBUG
			cudaStreamSynchronize(stream);
			cudaError_t error = cudaGetLastError();
			if( error != cudaSuccess ) {
				/*
				throw std::runtime_error(
					std::string("Transpose: CUDA error in kernel: ") +
					cudaGetErrorString(error));
				*/
			}
#endif
		}
	}
}
