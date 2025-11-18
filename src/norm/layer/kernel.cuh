#pragma once

#include "utils.cuh"
#include <type_traits>

constexpr size_t WARP_SIZE = 32;
constexpr float EPSILON = 1e-6f;

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

// One block processes one row.
// Shared memory size must be blockDim.x * sizeof(float).
template <typename T>
__global__ void layernorm_kernel_smem(T *output, const T *input,
                                      const float eps, size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const T *row_in = input + blockIdx.x * len;
    T *row_out = output + blockIdx.x * len;

    // ---- Pass 1: compute mean ----
    float sum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            sum += row_in[i];
        } else if constexpr (std::is_same_v<T, half>) {
            sum += __half2float(row_in[i]);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            sum += __bfloat162float(row_in[i]);
        }
    }

    smem[tid] = sum; // reduction buffer
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float mean = smem[0] / (float)len;

    // ---- Pass 2: compute variance ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        float d = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            d = row_in[i] - mean;
        } else if constexpr (std::is_same_v<T, half>) {
            d = __half2float(row_in[i]) - mean;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            d = __bfloat162float(row_in[i]) - mean;
        }
        sqsum += d * d;
    }

    smem[tid] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            row_out[i] = (row_in[i] - mean) * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            row_out[i] = __float2half((__half2float(row_in[i]) - mean) * inv_std);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            row_out[i] = __float2bfloat16((__bfloat162float(row_in[i]) - mean) * inv_std);
        }
    }
}

// One block processes one row.
// Shared memory size must be blockDim.x /  WARP_SIZE * sizeof(float).
template <typename T>
__global__ void layernorm_kernel_warp(T *output, const T *input,
                                      const float eps, size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f;
    const int warp_num = blockDim.x / WARP_SIZE;
    const T *row_in = input + blockIdx.x * len;
    T *row_out = output + blockIdx.x * len;

    // ---- Pass 1: compute mean ----
    float sum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            sum += row_in[i];
        } else if constexpr (std::is_same_v<T, half>) {
            sum += __half2float(row_in[i]);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            sum += __bfloat162float(row_in[i]);
        }
    }

    for (size_t stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }
    if (lane_id == 0) {
        smem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float block_sum = (tid < warp_num) ? smem[tid] : 0.0f;
        for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (tid == 0) {
            smem[tid] = block_sum;
        }
    }
    __syncthreads();

    float mean = smem[0] / (float)len;

    // ---- Pass 2: compute variance ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        float d = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            d = row_in[i] - mean;
        } else if constexpr (std::is_same_v<T, half>) {
            d = __half2float(row_in[i]) - mean;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            d = __bfloat162float(row_in[i]) - mean;
        }
        sqsum += d * d;
    }

    for (size_t stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        sqsum += __shfl_down_sync(0xffffffffu, sqsum, stride);
    }
    if (lane_id == 0) {
        smem[tid / WARP_SIZE] = sqsum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float block_sqsum = (tid < warp_num) ? smem[tid] : 0.0f;
        for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            block_sqsum += __shfl_down_sync(0xffffffffu, block_sqsum, offset);
        }
        if (tid == 0) {
            smem[tid] = block_sqsum;
        }
    }
    __syncthreads();

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            row_out[i] = (row_in[i] - mean) * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            row_out[i] = __float2half((__half2float(row_in[i]) - mean) * inv_std);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            row_out[i] = __float2bfloat16((__bfloat162float(row_in[i]) - mean) * inv_std);
        }
    }
}

// One block processes one row.
// Shared memory size must be blockDim.x * sizeof(float).
__global__ void layernorm_kernel_smem_vec(float *output, const float *input,
                                          const float eps, size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const float *row_in = input + blockIdx.x * len;
    float *row_out = output + blockIdx.x * len;

    // ---- Pass 1: compute mean ----
    float sum = 0.0f;
    size_t vec_len = len / 4;
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        const float4 v = FLOAT4_CONST(row_in[i * 4]);
        sum += v.x;
        sum += v.y;
        sum += v.z;
        sum += v.w;
    }

    smem[tid] = sum; // reduction buffer
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float mean = smem[0] / (float)len;

    // ---- Pass 2: compute variance ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        float d = 0.0f;
        const float4 v = FLOAT4_CONST(row_in[i * 4]);
        d = v.x - mean;
        sqsum += d * d;
        d = v.y - mean;
        sqsum += d * d;
        d = v.z - mean;
        sqsum += d * d;
        d = v.w - mean;
        sqsum += d * d;
    }

    smem[tid] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        float4 res;
        const float4 v = FLOAT4_CONST(row_in[i * 4]);
        res.x = (v.x - mean) * inv_std;
        res.y = (v.y - mean) * inv_std;
        res.z = (v.z - mean) * inv_std;
        res.w = (v.w - mean) * inv_std;
        FLOAT4(row_out[i * 4]) = res;
    }
}