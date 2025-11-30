#pragma once

#include "utils.cuh"
#include <mma.h>

using namespace nvcuda;

// Helper macros
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define CONST_FLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_fp16_kernel_v1(
    half *__restrict__ C,          // [M, N] row-major
    const half *__restrict__ A,    // [M, K] row-major
    const half *__restrict__ B,    // [N, K] row-major (transposed)
    const half *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores transposed B tile: [BK][BN] as B^T[k][n]
    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], __float2half(0.0f));
        }
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 128*32 = 4096 elements
    // Each thread loads 2 rows * 8 cols = 16 elements
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B (with transpose)
    // Need to load B[N,K] and store as s_b[K,N]
    // 256 threads load BK*BN = 32*256 = 8192 elements
    // Strategy: read B in N-contiguous manner, transpose to K-contiguous in smem
    // Each thread loads 4 k-values * 8 n-values = 32 elements
    int load_b_smem_k = (tid >> 5) << 2; // k: 0,4,8,...,28 (8 groups)
    int load_b_smem_n = (tid & 31) << 3; // n: 0,8,16,...,248 (32 threads per group)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    int num_k_tiles = ceil_div(K, BK);

    for (int bk = 0; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
// ==================== Load A tile ====================
// A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const bool k_aligned_8 = (K % 8 == 0);
            if (k_aligned_8 && gmem_m < M && gmem_k + 7 < K) {
                FLOAT4(s_a[smem_m][load_a_smem_k]) = CONST_FLOAT4(A[OFFSET(gmem_m, gmem_k, K)]);
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_m < M && gmem_k + j < K) {
                        s_a[smem_m][load_a_smem_k + j] = A[OFFSET(gmem_m, gmem_k + j, K)];
                    } else {
                        s_a[smem_m][load_a_smem_k + j] = __float2half(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile (with transpose) ====================
        // B[n][k] -> s_b[k][n], transpose during load
        // Read row-by-row from B (n-direction), store column-by-column to s_b (k-direction)
#pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            int smem_k = load_b_smem_k + ki;
            int gmem_k = k_start + smem_k;

#pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int smem_n = load_b_smem_n + ni;
                int gmem_n = load_b_gmem_n + ni;

                if (gmem_n < N && gmem_k < K) {
                    // B[n,k] stored at B_ptr[n * K + k]
                    // Store to s_b[k][n] for transposed access
                    s_b[smem_k][smem_n] = B[OFFSET(gmem_n, gmem_k, K)];
                } else {
                    s_b[smem_k][smem_n] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major
        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        // Load B fragments: s_b[k][n] with row_major (already transposed)
        wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;

    // Temp buffer for output
    __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store fragment to shared memory
            wmma::store_matrix_sync(&s_c_float[wid][0][0], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

// Add bias and write to global memory
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                int global_m = tile_m + local_m;
                int global_n = tile_n + local_n;

                if (global_m < M && global_n < N) {
                    float val = s_c_float[wid][local_m][local_n];
                    float b = __half2float(bias[global_n]);
                    C[OFFSET(global_m, global_n, N)] = __float2half(val + b);
                }
            }
            __syncwarp();
        }
    }
}

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_fp16_kernel_v2(
    half *__restrict__ C,          // [M, N] row-major
    const half *__restrict__ A,    // [M, K] row-major
    const half *__restrict__ B,    // [N, K] row-major (transposed)
    const half *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BN][BK + BPAD];

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 128*32 = 4096 elements
    // Each thread loads 2 rows * 8 cols = 16 elements
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 256*32 = 8192 elements
    // Each thread loads 4 rows * 8 cols = 32 elements
    int load_b_smem_n = (tid >> 2) << 2;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    int num_k_tiles = ceil_div(K, BK);

    for (int bk = 0; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
// ==================== Load A tile ====================
// A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const bool k_aligned_8 = (K % 8 == 0);
            if (k_aligned_8 && gmem_m < M && gmem_k + 7 < K) {
                FLOAT4(s_a[smem_m][load_a_smem_k]) = CONST_FLOAT4(A[OFFSET(gmem_m, gmem_k, K)]);
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_m < M && gmem_k + j < K) {
                        s_a[smem_m][load_a_smem_k + j] = A[OFFSET(gmem_m, gmem_k + j, K)];
                    } else {
                        s_a[smem_m][load_a_smem_k + j] = __float2half(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = k_start + load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const bool k_aligned_8 = (K % 8 == 0);
            if (k_aligned_8 && gmem_n < N && gmem_k + 7 < K) {
                FLOAT4(s_b[smem_n][load_b_smem_k]) = CONST_FLOAT4(B[OFFSET(gmem_n, gmem_k, K)]);
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_n < N && gmem_k + j < K) {
                        s_b[smem_n][load_b_smem_k + j] = B[OFFSET(gmem_n, gmem_k + j, K)];
                    } else {
                        s_b[smem_n][load_b_smem_k + j] = __float2half(0.0f);
                    }
                }
            }
        }

        __syncthreads();

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major
        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major
        wmma::load_matrix_sync(frag_b[0][0], &s_b[comp_c_frag_n * 64][0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[comp_c_frag_n * 64 + 16][0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[comp_c_frag_n * 64 + 32][0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[comp_c_frag_n * 64 + 48][0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[comp_c_frag_n * 64][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[comp_c_frag_n * 64 + 16][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[comp_c_frag_n * 64 + 32][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[comp_c_frag_n * 64 + 48][16], BK + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;

    // Temp buffer for output
    __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    __shared__ half s_c_half[8][16][16];   // 8 warps, 16x16 each

    // Each warp handles a 64x64 output block, divided into 4x4 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15
    //
    // For the entire 64x64 block, each thread writes:
    //   - 4 (i-tiles) × 4 (j-tiles) × 8 (rows per tile) = 128 elements
    //   - Spanning 32 rows (8 rows × 4 i-tiles) × 4 columns (one per j-tile)
    //
    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 4 bias values (one for each j-tile).
    float bias_vals[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N) ? __half2float(bias[global_n]) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store fragment to shared memory
            wmma::store_matrix_sync(&s_c_float[wid][0][0], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add bias
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                int global_m = tile_m + local_m;
                int global_n = tile_n + local_n;

                s_c_half[wid][local_m][local_n] = __float2half(s_c_float[wid][local_m][local_n] + bias_vals[j]);
            }
            __syncwarp();

            // Write to global memory using FLOAT4 (128-bit = 8 halfs)
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized 128-bit store (8 halfs at once)
                FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_half[wid][row][col]);
            } else if (global_m < M) {
                // Boundary case: element-wise store
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_half[wid][row][col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}