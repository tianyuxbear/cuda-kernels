#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <vector>

void max_cpu(float *output, const std::vector<float> &input) {
  for (size_t i = 0; i < input.size(); ++i) {
    *output = std::max(*output, input[i]);
  }
}

#define TEST1

#ifdef TEST1
#define test_kernel reduce_max_kernel_warp
#define config_name "reduce_max_warp"
#endif

#ifdef TEST2
#define test_kernel reduce_max_kernel_warp_smem
#define config_name "reduce_max_warp_smem"
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
  std::vector<float> h_input(N);
  float h_output{};
  float h_output_gpu{};

  // Allocate device memory
  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, nbytes));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

  // Copy data to device
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), nbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice));

  // Set up kernel configuration
  dim3 blockDim(BLOCK_SIZE);
  //   dim3 gridDim(ceil_div(N, BLOCK_SIZE));
  dim3 gridDim(BLOCK_SIZE);

  // Warm-up iterations
  for (int i = 0; i < WARM_UP_ITERS; ++i) {
    test_kernel<<<gridDim, blockDim>>>(d_output, d_input, N);
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
    test_kernel<<<gridDim, blockDim>>>(d_output, d_input, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_time_ms += elapsed_ms;
  }

  // Calculate average kernel time
  float avg_kernel_time_ms = total_time_ms / PROFILE_ITERS;

  // Calculate performance metrics
  float memory_usage_mb = (1.0 * nbytes) / (1024.0 * 1024.0);

  size_t bytes_accessed = 1 * N * sizeof(float);
  float throughput_gbs = (bytes_accessed / (avg_kernel_time_ms / 1000.0)) / 1e9;

  size_t flops = N;
  float compute_perf_gflops = (flops / (avg_kernel_time_ms / 1000.0)) / 1e9;

  // Copy the result back to host
  CUDA_CHECK(cudaMemcpy(&h_output_gpu, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU reference calculation
  max_cpu(&h_output, h_input);

  // Verify results
  bool verification_passed = verify_results(&h_output, &h_output_gpu, 1);

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
  if (d_output)
    CUDA_CHECK(cudaFree(d_output));
  if (d_input)
    CUDA_CHECK(cudaFree(d_input));
}