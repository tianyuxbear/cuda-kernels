#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

void layernorm_cpu(std::vector<float> &output, const std::vector<float> &input,
                   const float eps, const size_t nrow, const size_t len) {
    if (len == 0) {
        throw std::invalid_argument("len must be > 0");
    }
    const size_t total = nrow * len;
    if (input.size() != total) {
        throw std::invalid_argument("input.size() must equal nrow * len");
    }
    if (output.size() != total) {
        throw std::invalid_argument("output.size() must equal nrow * len");
    }

    for (size_t r = 0; r < nrow; ++r) {
        const size_t base = r * len;

        float sum = 0.0;
        for (size_t i = 0; i < len; ++i) {
            sum += input[base + i];
        }
        float mean = sum / (float)len;

        // compute variance (population variance: divide by len)
        float sqsum = 0.0;
        for (size_t i = 0; i < len; ++i) {
            float d = input[base + i] - mean;
            sqsum += d * d;
        }
        float var = sqsum / (float)len;
        float inv_std = 1.0 / std::sqrt(var + eps);

        // normalize and write back
        for (size_t i = 0; i < len; ++i) {
            float normalized = (input[base + i] - mean) * inv_std;
            output[base + i] = normalized;
        }
    }
}

#define TEST3

#ifdef TEST1
#define test_kernel layernorm_kernel_smem
#define config_name "layernorm_smem"
#endif

#ifdef TEST2
#define test_kernel layernorm_kernel_warp
#define config_name "layernorm_warp"
#endif

#ifdef TEST3
#define test_kernel layernorm_kernel_smem_vec
#define config_name "layernorm_smem_vec"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

int main(int argc, char *argv[]) {
    constexpr size_t len = 1536;
    size_t nrow;
    if (argc == 2) {
        nrow = 1ul << std::atoi(argv[1]);
    } else {
        nrow = 128;
    }

    const size_t nbytes = nrow * len * sizeof(float);

    // Initialize data
    std::vector<float> h_input(nrow * len);
    std::vector<float> h_output(nrow * len, 0);
    std::vector<float> h_output_gpu(nrow * len, 0);
    fill_array_with_random_floats(h_input.data(), h_input.size());

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, nbytes));
    CUDA_CHECK(cudaMalloc(&d_output, nbytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, h_output.data(), nbytes, cudaMemcpyHostToDevice));

    // Set up kernel configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(nrow);

#if defined(TEST1) || defined(TEST3)
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
#elif defined(TEST2)
    size_t shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);
#endif

    // Warm-up iterations
    for (int i = 0; i < WARM_UP_ITERS; ++i) {
        test_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_output, d_input, EPSILON, len);
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
        test_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_output, d_input, EPSILON, len);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Calculate average kernel time
    float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

    // Calculate performance metrics
    float memory_usage_mb = (2.0 * nbytes) / (1024.0 * 1024.0);

    size_t bytes_accessed = 4 * nbytes;
    float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

    size_t flops = 5 * nrow * len;
    float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, nbytes, cudaMemcpyDeviceToHost));

    // CPU reference calculation
    layernorm_cpu(h_output, h_input, EPSILON, nrow, len);

    // Verify results
    bool verification_passed = verify_results(h_output.data(), h_output_gpu.data(), nrow * len, 1e-5f);

    BenchmarkResults results{config_name,
                             nrow * len,
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
    if (d_input) {
        CUDA_CHECK(cudaFree(d_input));
    }
    if (d_output) {
        CUDA_CHECK(cudaFree(d_output));
    }
}