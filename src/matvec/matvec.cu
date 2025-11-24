#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <vector>

void matvec_cpu(std::vector<float> &c, const std::vector<float> &a,
                const std::vector<float> &b, const std::vector<float> &bias, const size_t N, const size_t K) {
    for (size_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        const float *b_ptr = b.data() + i * K;
        for (size_t p = 0; p < K; ++p) {
            sum += a[p] * b_ptr[p];
        }
        sum += bias[i];
        c[i] = sum;
    }
}

#define TEST3

#ifdef TEST1
#define test_kernel matvec_kernel_naive
#define config_name "matvec_naive"
#endif

#ifdef TEST2
#define test_kernel matvec_kernel_warp
#define config_name "matvec_warp"
#endif

#ifdef TEST3
#define test_kernel matvec_kernel_warp_vec
#define config_name "matvec_warp_vec"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

int main(int argc, char *argv[]) {
    size_t N, K;
    if (argc == 3) {
        N = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
    } else {
        N = 1024;
        K = 1024;
    }

    // Initialize data
    std::vector<float> h_a(K);
    std::vector<float> h_b(N * K);
    std::vector<float> h_bias(N);
    std::vector<float> h_c(N, 0);
    std::vector<float> h_c_gpu(N, 0);
    fill_array_with_random_floats(h_a.data(), h_a.size());
    fill_array_with_random_floats(h_b.data(), h_b.size());
    fill_array_with_random_floats(h_bias.data(), h_bias.size());

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_a, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Set up kernel configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil_div(N, BLOCK_SIZE));

#if defined(TEST2)
    gridDim.x = BLOCK_SIZE;
#endif

#if defined(TEST3)
    gridDim.x = N;
#endif

    // Warm-up iterations
    for (int i = 0; i < WARM_UP_ITERS; ++i) {
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, N, K);
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
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, d_bias, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Calculate average kernel time
    float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

    // Calculate performance metrics
    float memory_usage_mb = (K + N * K + N + N) * sizeof(float) / (1024.0 * 1024.0);

    size_t bytes_accessed = (K + N * K + N + N) * sizeof(float);
    float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

    size_t flops = 2 * N * K;
    float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference calculation
    matvec_cpu(h_c, h_a, h_b, h_bias, N, K);

    // Verify results
    bool verification_passed = verify_results(h_c.data(), h_c_gpu.data(), N);

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
    if (d_bias) {
        CUDA_CHECK(cudaFree(d_bias));
    }
}