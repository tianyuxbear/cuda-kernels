#include "check.cuh"
#include "linear_fp32_kernel.cuh"
#include "utils.cuh"

#include <cstddef>
#include <vector>

#include <cstddef>
#include <vector>

void linear_cpu(std::vector<float> &c,
                const std::vector<float> &a,
                const std::vector<float> &b,
                const std::vector<float> &bias,
                size_t M, size_t N, size_t K) {
    const float *__restrict a_ptr = a.data();
    const float *__restrict b_ptr = b.data();
    const float *__restrict bias_ptr = bias.data();
    float *__restrict c_ptr = c.data();

    for (size_t i = 0; i < M; ++i) {
        const float *a_row = a_ptr + i * K;
        float *c_row = c_ptr + i * N;

        for (size_t j = 0; j < N; ++j) {
            const float *b_row = b_ptr + j * K;

            float sum = bias_ptr[j];

            for (size_t p = 0; p < K; ++p) {
                sum += a_row[p] * b_row[p];
            }

            c_row[j] = sum;
        }
    }
}

#define TEST1

#ifdef TEST1
#define test_kernel linear_fp32_kernel
#define config_name "linear_fp32"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

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
    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N);
    std::vector<float> h_bias(N);
    std::vector<float> h_c_gpu(M * N, 0);
    fill_array_with_random_floats(h_a.data(), h_a.size(), -1.0f, 1.0f);
    fill_array_with_random_floats(h_b.data(), h_b.size(), -1.0f, 1.0f);
    fill_array_with_random_floats(h_c.data(), h_c.size(), -1.0f, 1.0f);
    fill_array_with_random_floats(h_bias.data(), h_bias.size(), -1.0f, 1.0f);

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Set up kernel configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim;
    gridDim.y = ceil_div(M, 128);
    gridDim.x = ceil_div(N, 128);

    // Warm-up iterations
    for (int i = 0; i < WARM_UP_ITERS; ++i) {
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, M, N, K);
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
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Calculate average kernel time
    float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

    // Calculate performance metrics
    float memory_usage_mb = (M * K + K * N + M * N) * sizeof(float) / (1024.0 * 1024.0);

    size_t bytes_accessed = (M * K + K * N + M * N) * sizeof(float);
    float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

    size_t flops = 2 * M * N * K;
    float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference calculation
    linear_cpu(h_c, h_a, h_b, h_bias, M, N, K);

    // Verify results
    bool verification_passed = verify_results(h_c.data(), h_c_gpu.data(), M * N, 1e-4f, 1e-5f);
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