#pragma once

#include "utils.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T> __device__ __forceinline__ T get_lowest_value() {
  if constexpr (std::is_same_v<T, float>) {
    return -CUDART_INF_F;
  } else if (std::is_same_v<T, half>) {
    return __float2half(-65504.0f);
  } else if (std::is_same_v<T, cuda_bfloat16>) {
    return __float2bfloat16(-CUDART_INF_F);
  }
}

template <typename T>
__global__ void reduce_argmax_kernel_block(size_t *max_idx, T *max_val,
                                           const T *input, size_t N) {
  __shared__ T s_val[BLOCK_SIZE * 4];
  __shared__ size_t s_idx[BLOCK_SIZE * 4];

  size_t tid = threadIdx.x;
  size_t idx = tid;

  T my_val = get_lowest_value<T>();
  size_t my_idx = 0;

  // Each thread process multiple elements
  while (idx < N) {
    if (input[idx] > my_val) {
      my_val = input[idx];
      my_idx = idx;
    }
    idx += blockDim.x;
  }

  s_val[tid] = my_val;
  s_idx[tid] = my_idx;
  __syncthreads();

  // block reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_val[tid + stride] > s_val[tid]) {
        s_val[tid] = s_val[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }

  // only one writer -> no need for atomic
  if (tid == 0) {
    *max_val = s_val[0];
    *max_idx = s_idx[0];
  }
}