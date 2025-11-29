#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define OFFSET(row, col, stride) ((row) * (stride) + (col))

/**
 * 验证方法：用单位矩阵 A = I 做乘法
 * C = A * B = I * B = B
 * 这样 C 的结果就直接反映了 B fragment 的加载是否正确
 *
 * 期望：C[m][n] = B[n][m]（因为 B 是转置存储的）
 * 即：C[m][n] = sum_k(A[m][k] * B_transposed[k][n]) = sum_k(I[m][k] * B[n][k]) = B[n][m]
 */

// 方法1: col_major 加载 B（不转置）
__global__ void test_col_major_gemm(
    const half *__restrict__ A, // [16, 16] 单位矩阵
    const half *__restrict__ B, // [16, 16] row-major，存储的是 B^T
    half *__restrict__ C,
    int M, int N, int K) {

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[16][16 + APAD]; // [M][K]
    __shared__ half s_b[16][16 + BPAD]; // [N][K] for col_major

    int tid = threadIdx.x;

    // 加载 A (单位矩阵)
    if (tid < 16) {
        for (int k = 0; k < 16; k++) {
            s_a[tid][k] = A[OFFSET(tid, k, K)];
        }
    }

    // 加载 B：s_b[n][k] = B[n][k]（不转置）
    if (tid < 16) {
        for (int k = 0; k < 16; k++) {
            s_b[tid][k] = B[OFFSET(tid, k, K)];
        }
    }
    __syncthreads();

    // WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    // 加载 A: row_major, stride = 16 + APAD = 24
    wmma::load_matrix_sync(frag_a, &s_a[0][0], 16 + APAD);

    // 加载 B: col_major, s_b[n][k]，stride = 16 + BPAD = 24
    // col_major: B_frag[k][n] = s_b[n * stride + k] = s_b[n][k]
    wmma::load_matrix_sync(frag_b, &s_b[0][0], 16 + BPAD);

    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    // 存储结果
    wmma::store_matrix_sync(C, frag_c, N, wmma::mem_row_major);
}

// 方法2: row_major 加载 B（需要转置）
__global__ void test_row_major_gemm(
    const half *__restrict__ A,
    const half *__restrict__ B,
    half *__restrict__ C,
    int M, int N, int K) {

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[16][16 + APAD]; // [M][K]
    __shared__ half s_b[16][16 + BPAD]; // [K][N] for row_major

    int tid = threadIdx.x;

    // 加载 A
    if (tid < 16) {
        for (int k = 0; k < 16; k++) {
            s_a[tid][k] = A[OFFSET(tid, k, K)];
        }
    }

    // 加载 B 并转置：s_b[k][n] = B[n][k]
    if (tid < 16) {
        for (int k = 0; k < 16; k++) {
            s_b[k][tid] = B[OFFSET(tid, k, K)]; // tid = n, 转置！
        }
    }
    __syncthreads();

    // WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    wmma::load_matrix_sync(frag_a, &s_a[0][0], 16 + APAD);

    // row_major: B_frag[k][n] = s_b[k * stride + n] = s_b[k][n]
    wmma::load_matrix_sync(frag_b, &s_b[0][0], 16 + BPAD);

    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    wmma::store_matrix_sync(C, frag_c, N, wmma::mem_row_major);
}

void print_matrix(half *data, int rows, int cols, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 8; i++) {
        for (int j = 0; j < cols && j < 8; j++) {
            printf("%7.1f ", __half2float(data[i * cols + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int M = 16, N = 16, K = 16;

    // A = 单位矩阵
    half *h_A = new half[M * K];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half(i == j ? 1.0f : 0.0f);
        }
    }

    // B[n][k] = n * 100 + k (转置存储，即实际的 B^T)
    // 这意味着逻辑上的 B_logical[k][n] = B_stored[n][k] = n * 100 + k
    half *h_B = new half[N * K];
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            h_B[n * K + k] = __float2half((float)(n * 100 + k));
        }
    }

    printf("=== Input ===\n");
    printf("A = Identity matrix (16x16)\n");
    printf("B stored as [N][K], B[n][k] = n*100 + k\n");
    printf("Logical B^T[k][n] = B[n][k] = n*100 + k\n\n");

    printf("B (first 4x4):\n");
    for (int n = 0; n < 4; n++) {
        for (int k = 0; k < 4; k++) {
            printf("%7.1f ", __half2float(h_B[n * K + k]));
        }
        printf("   (n=%d)\n", n);
    }
    printf("\n");

    printf("Expected C = A * B^T = I * B^T:\n");
    printf("C[m][n] = B^T[m][n] = B[n][m] = n*100 + m\n");
    printf("So C[0][0]=0, C[0][1]=100, C[0][2]=200, C[1][0]=1, etc.\n\n");

    printf("Expected C (first 4x4):\n");
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            printf("%7.1f ", (float)(n * 100 + m));
        }
        printf("\n");
    }
    printf("\n");

    half *d_A, *d_B, *d_C1, *d_C2;
    half *h_C1 = new half[M * N];
    half *h_C2 = new half[M * N];

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K * sizeof(half));
    cudaMalloc(&d_C1, M * N * sizeof(half));
    cudaMalloc(&d_C2, M * N * sizeof(half));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice);

    test_col_major_gemm<<<1, 256>>>(d_A, d_B, d_C1, M, N, K);
    test_row_major_gemm<<<1, 256>>>(d_A, d_B, d_C2, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_C1, d_C1, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    printf("=== Results ===\n");
    print_matrix(h_C1, M, N, "Col-major method C");
    print_matrix(h_C2, M, N, "Row-major (transpose) method C");

    // 验证
    int errors1 = 0, errors2 = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = (float)(n * 100 + m);
            float got1 = __half2float(h_C1[m * N + n]);
            float got2 = __half2float(h_C2[m * N + n]);
            if (fabs(got1 - expected) > 0.1f) {
                errors1++;
            }
            if (fabs(got2 - expected) > 0.1f) {
                errors2++;
            }
        }
    }

    printf("Col-major method errors: %d / %d\n", errors1, M * N);
    printf("Row-major method errors: %d / %d\n", errors2, M * N);

    if (errors1 == 0) {
        printf("\n✓ Col-major method is CORRECT!\n");
    } else {
        printf("\n✗ Col-major method is WRONG!\n");
    }

    if (errors2 == 0) {
        printf("✓ Row-major method is CORRECT!\n");
    } else {
        printf("✗ Row-major method is WRONG!\n");
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C1;
    delete[] h_C2;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);

    return 0;
}
