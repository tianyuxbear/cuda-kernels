#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <vector>

void add_cpu(std::vector<float> &c, const std::vector<float> &a,
             const std::vector<float> &b) {
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

#define TEST1

#ifdef TEST1
#define test_kernel add_kernel
#define config_name "add"
#endif

#ifdef TEST2
#define test_kernel add_f32x4_kernel
#define config_name "add_f32x4"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

int main(int argc, char *argv[]) {
    size_t N;
    if (argc == 2) {
        N = 1ul << std::atoi(argv[1]);
    } else {
        N = 1ul << 20;
    }

    const size_t nbytes = N * sizeof(float);

    // Initialize data
    std::vector<float> h_a(N, 1);
    std::vector<float> h_b(N, 2);
    std::vector<float> h_c(N, 0);
    std::vector<float> h_c_gpu(N, 0);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, nbytes));
    CUDA_CHECK(cudaMalloc(&d_b, nbytes));
    CUDA_CHECK(cudaMalloc(&d_c, nbytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), nbytes, cudaMemcpyHostToDevice));

    // Set up kernel configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil_div(N, BLOCK_SIZE));

    // Warm-up iterations
    for (int i = 0; i < WARM_UP_ITERS; ++i) {
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, N);
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
        test_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Calculate average kernel time
    float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

    // Calculate performance metrics
    float memory_usage_mb = (3.0 * nbytes) / (1024.0 * 1024.0);

    size_t bytes_accessed = 3 * N * sizeof(float);
    float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

    size_t flops = N;
    float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, nbytes, cudaMemcpyDeviceToHost));

    // CPU reference calculation
    add_cpu(h_c, h_a, h_b);

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
}