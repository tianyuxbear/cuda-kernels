#pragma once

#include "utils.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
__device__ __forceinline__ T get_lowest_value() {
    if constexpr (std::is_same_v<T, float>) {
        return -CUDART_INF_F;
    } else if (std::is_same_v<T, half>) {
        return __float2half(-65504.0f);
    } else if (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(-CUDART_INF_F);
    }
}

// AtomicMax for float
__device__ __forceinline__ float atomicMaxFloat(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        // Convert to float, compare, and convert back to int
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);

    return __int_as_float(old);
}

// AtomicMax for __half (FP16)
__device__ __forceinline__ __half atomicMaxHalf(__half *address, __half val) {
    // Check if address is 4-byte aligned
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int offset = (size_t)address & 3;
    unsigned int shift = (offset >> 1) * 16; // 0 or 16 bits

    unsigned int old = *base_address;
    unsigned int assumed;
    unsigned short old_half, new_half;

    do {
        assumed = old;
        old_half = (unsigned short)((old >> shift) & 0xffff);
        __half old_val = __ushort_as_half(old_half);
        __half new_val = __hmax(old_val, val);
        new_half = __half_as_ushort(new_val);

        unsigned int new_full = (old & ~(0xffff << shift)) | (new_half << shift);
        old = atomicCAS(base_address, assumed, new_full);
    } while (assumed != old);

    return __ushort_as_half((unsigned short)((old >> shift) & 0xffff));
}

// AtomicMax for __nv_bfloat16 (BF16)
__device__ __forceinline__ __nv_bfloat16
atomicMaxBfloat16(__nv_bfloat16 *address, __nv_bfloat16 val) {
    // Check if address is 4-byte aligned
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int offset = (size_t)address & 3;
    unsigned int shift = (offset >> 1) * 16; // 0 or 16 bits

    unsigned int old = *base_address;
    unsigned int assumed;
    unsigned short old_bf16, new_bf16;

    do {
        assumed = old;
        old_bf16 = (unsigned short)((old >> shift) & 0xffff);
        __nv_bfloat16 old_val = __ushort_as_bfloat16(old_bf16);
        __nv_bfloat16 new_val = __hmax(old_val, val);
        new_bf16 = __bfloat16_as_ushort(new_val);

        unsigned int new_full = (old & ~(0xffff << shift)) | (new_bf16 << shift);
        old = atomicCAS(base_address, assumed, new_full);
    } while (assumed != old);

    return __ushort_as_bfloat16((unsigned short)((old >> shift) & 0xffff));
}

// Template wrapper
template <typename T>
__device__ __forceinline__ T atomicMaxAny(T *address, T val);

template <>
__device__ __forceinline__ float atomicMaxAny<float>(float *address,
                                                     float val) {
    return atomicMaxFloat(address, val);
}

template <>
__device__ __forceinline__ __half atomicMaxAny<__half>(__half *address,
                                                       __half val) {
    return atomicMaxHalf(address, val);
}

template <>
__device__ __forceinline__ __nv_bfloat16
atomicMaxAny<__nv_bfloat16>(__nv_bfloat16 *address, __nv_bfloat16 val) {
    return atomicMaxBfloat16(address, val);
}

template <typename T>
__global__ void reduce_max_kernel_warp(T *output, const T *input, size_t N) {
    size_t tid = threadIdx.x;
    size_t lane_id = tid % 32;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize to the minimum value of type T
    T thread_max = get_lowest_value<T>();

    // Each thread finds the maximum value among its assigned elements
    for (size_t j = idx; j < N; j += blockDim.x * gridDim.x) {
        thread_max = max(thread_max, input[j]);
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        T other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = max(thread_max, other_max);
    }

    // Lane 0 of each warp atomically updates the global result
    if (lane_id == 0) {
        atomicMaxAny(output, thread_max);
    }
}

template <typename T>
__device__ T warp_reduce(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

template <typename T>
__global__ void reduce_max_kernel_warp_smem(T *output, const T *input,
                                            size_t N) {
    __shared__ T smem[32];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Initialize to the minimum value of type T
    T thread_max = get_lowest_value<T>();

    // Each thread finds the maximum value among its assigned elements
    for (size_t j = idx; j < N; j += blockDim.x * gridDim.x) {
        thread_max = max(thread_max, input[j]);
    }

    // Warp-level reduction using shuffle instructions
    T warp_max = warp_reduce(thread_max);

    if (tid % 32 == 0) {
        smem[tid / 32] = warp_max;
    }

    if (tid < 32) {
        T block_max = (tid < (blockDim.x + 31) / 32) ? smem[tid] : T(0);
        block_max = warp_reduce(block_max);
        if (tid == 0) {
            atomicMaxAny(output, block_max);
        }
    }
}
