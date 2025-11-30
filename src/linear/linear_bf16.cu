#include "check.cuh"
#include "linear_bf16_kernel.cuh"
#include "utils.cuh"

#include <cstddef>

void linear_cpu_bf16(cuda_bfloat16 *c,
                     const cuda_bfloat16 *a,
                     const cuda_bfloat16 *b,
                     const cuda_bfloat16 *bias,
                     size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        const cuda_bfloat16 *a_row = a + i * K;
        cuda_bfloat16 *c_row = c + i * N;

        for (size_t j = 0; j < N; ++j) {
            const cuda_bfloat16 *b_row = b + j * K;

            float sum = __bfloat162float(bias[j]);

            for (size_t p = 0; p < K; ++p) {
                sum += __bfloat162float(a_row[p]) * __bfloat162float(b_row[p]);
            }

            c_row[j] = __float2bfloat16(sum);
        }
    }
}

#define TEST4

#ifdef TEST1
#define test_kernel linear_bf16_kernel_v1
#define config_name "linear_bf16_v1"
#endif

#ifdef TEST2
#define test_kernel linear_bf16_kernel_v2
#define config_name "linear_bf16_v2"
#endif

#ifdef TEST3
#define test_kernel linear_bf16_kernel_v3
#define config_name "linear_bf16_v3"
#endif

#ifdef TEST4
#define test_kernel linear_bf16_kernel_v4
#define config_name "linear_bf16_v4"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

constexpr size_t BM = 128;
constexpr size_t BN = 256;

int main(int argc, char *argv[]) {
    size_t M, N, K;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        M = 1024;
        N = 1024;
        K = 1024;
    }
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    // Initialize data
    size_t size_a = M * K * sizeof(cuda_bfloat16);
    size_t size_b = K * N * sizeof(cuda_bfloat16);
    size_t size_c = M * N * sizeof(cuda_bfloat16);
    size_t size_bias = N * sizeof(cuda_bfloat16);

    cuda_bfloat16 *h_a = (cuda_bfloat16 *)malloc(size_a);
    cuda_bfloat16 *h_b = (cuda_bfloat16 *)malloc(size_b);
    cuda_bfloat16 *h_c = (cuda_bfloat16 *)malloc(size_c);
    cuda_bfloat16 *h_bias = (cuda_bfloat16 *)malloc(size_bias);
    cuda_bfloat16 *h_c_gpu = (cuda_bfloat16 *)malloc(size_c);
    memset(h_c_gpu, 0, size_c);

    fill_array_with_random_bf16s(h_a, M * K, -1.0f, 1.0f);
    fill_array_with_random_bf16s(h_b, K * N, -1.0f, 1.0f);
    fill_array_with_random_bf16s(h_c, M * N, -1.0f, 1.0f);
    fill_array_with_random_bf16s(h_bias, N, -1.0f, 1.0f);

    // Allocate device memory
    cuda_bfloat16 *d_a, *d_b, *d_c, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, size_c, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, size_bias, cudaMemcpyHostToDevice));

    // Set up kernel configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim;
    gridDim.y = ceil_div(M, BM);
    gridDim.x = ceil_div(N, BN);

#ifdef TEST4
    constexpr size_t BK = 32;
    constexpr size_t PAD = 8;
    constexpr size_t smem_size = 2 * (BM + BN) * (BK + PAD) * sizeof(cuda_bfloat16) + 8 * 16 * 16 * sizeof(float) + 8 * 16 * 16 * sizeof(cuda_bfloat16);
    cudaFuncSetAttribute(linear_bf16_kernel_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
#endif

    // Warm-up iterations
    for (int i = 0; i < WARM_UP_ITERS; ++i) {
#ifdef TEST4
        test_kernel<<<gridDim, blockDim, smem_size>>>(d_c, d_a, d_b, d_bias, M, N, K);
#else
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, M, N, K);
#endif
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Profile iterations with timing
    float total_time_ms = 0.0f;
    for (int i = 0; i < PROFILE_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
#ifdef TEST4
        test_kernel<<<gridDim, blockDim, smem_size>>>(d_c, d_a, d_b, d_bias, M, N, K);
#else
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, M, N, K);
#endif
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Calculate average kernel time
    float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

    // Calculate performance metrics
    float memory_usage_mb = (M * K + K * N + M * N) * sizeof(cuda_bfloat16) / (1024.0 * 1024.0);

    size_t bytes_accessed = (M * K + K * N + M * N) * sizeof(cuda_bfloat16);
    float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

    size_t flops = 2 * M * N * K;
    float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost));

    // CPU reference calculation
    linear_cpu_bf16(h_c, h_a, h_b, h_bias, M, N, K);

    // Verify results
    bool verification_passed = verify_results_bf16(h_c, h_c_gpu, M * N, 1e-2f, 1e-2f);
    BenchmarkResults results{config_name,
                             N,
                             memory_usage_mb,
                             blockDim,
                             gridDim,
                             WARM_UP_ITERS,
                             PROFILE_ITERS,
                             avg_kernel_time_ms,
                             throughput_gbs,
                             compute_perf_gflops,
                             verification_passed};

    print_benchmark_results(results);

    // Release device memory
    if (d_a) {
        CUDA_CHECK(cudaFree(d_a));
    }
    if (d_b) {
        CUDA_CHECK(cudaFree(d_b));
    }
    if (d_c) {
        CUDA_CHECK(cudaFree(d_c));
    }
}