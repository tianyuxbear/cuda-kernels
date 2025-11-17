#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <cstring>
#include <vector>

void embedding_cpu(std::vector<float> &output, const std::vector<float> &weight,
                   const std::vector<int> &index, size_t len) {
  for (size_t i = 0; i < index.size(); ++i) {
    const float *src = weight.data() + index[i] * len;
    float *dst = output.data() + i * len;
    std::memcpy(dst, src, sizeof(float) * len);
  }
}

#define TEST1

#ifdef TEST1
#define test_kernel embedding_kernel
#define config_name "embedding_naive"
#endif

#ifdef TEST2
#define test_kernel embedding_kernel_vec
#define config_name "embedding_vec"
#endif

constexpr size_t WARM_UP_ITERS = 10;
constexpr size_t PROFILE_ITERS = 10;

int main(int argc, char *argv[]) {
  constexpr size_t M = 151936;
  constexpr size_t K = 1536;

  size_t N;
  if (argc == 2) {
    N = std::atoi(argv[1]);
  } else {
    N = 128;
  }

  // Initialize data
  std::vector<float> h_weight(M * K);
  std::vector<float> h_output(N * K, 0);
  std::vector<float> h_output_gpu(N * K, 0);
  fill_array_with_random_floats(h_weight.data(), h_weight.size());

  std::vector<int> h_index = random_sample_ints(M, N);

  // Allocate device memory
  float *d_weight, *d_output;
  int *d_index;
  CUDA_CHECK(cudaMalloc(&d_weight, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, N * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_index, N * sizeof(int)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output, h_output.data(), N * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_index, h_index.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Set up kernel configuration
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(N);

  // Warm-up iterations
  for (int i = 0; i < WARM_UP_ITERS; ++i) {
    test_kernel<<<gridDim, blockDim>>>(d_output, d_weight, d_index, K);
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
    test_kernel<<<gridDim, blockDim>>>(d_output, d_weight, d_index, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_time_ms += elapsed_ms;
  }

  // Calculate average kernel time
  float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

  // Calculate performance metrics
  float memory_usage_mb =
      ((M * K + N * K) * sizeof(float) + N * sizeof(int)) / (1024.0 * 1024.0);

  size_t bytes_accessed = 2 * N * K * sizeof(float) + N * sizeof(int);
  float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

  size_t flops = 2 * N * K;
  float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

  // Copy the result back to host
  CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, N * K * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU reference calculation
  embedding_cpu(h_output, h_weight, h_index, K);

  // Verify results
  bool verification_passed =
      verify_results(h_output.data(), h_output_gpu.data(), N * K);

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
  if (d_weight)
    CUDA_CHECK(cudaFree(d_weight));
  if (d_output)
    CUDA_CHECK(cudaFree(d_output));
  if (d_index)
    CUDA_CHECK(cudaFree(d_index));
}