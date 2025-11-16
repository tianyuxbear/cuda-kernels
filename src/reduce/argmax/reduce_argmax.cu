#include "check.cuh"
#include "kernel.cuh"
#include "utils.cuh"

#include <vector>

void argmax_cpu(size_t *max_idx, float *max_val,
                const std::vector<float> &input) {
  *max_val = input[0];
  *max_idx = 0;
  for (size_t i = 1; i < input.size(); ++i) {
    if (input[i] > *max_val) {
      *max_val = input[i];
      *max_idx = i;
    }
  }
}

#define TEST0

#ifdef TEST0
#define test_kernel reduce_argmax_kernel_block
#define config_name "reduce_argmax_block"
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
  size_t h_max_idx{}, h_max_idx_gpu{};
  float h_max_val{}, h_max_val_gpu{};

  // Allocate device memory
  float *d_input, *d_max_val;
  size_t *d_max_idx;
  CUDA_CHECK(cudaMalloc(&d_input, nbytes));
  CUDA_CHECK(cudaMalloc(&d_max_idx, sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(float)));

  // Copy data to device
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), nbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_max_idx, &h_max_idx, sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_max_val, &h_max_val, sizeof(float), cudaMemcpyHostToDevice));

  // Set up kernel configuration
  dim3 blockDim(BLOCK_SIZE * 4);
  //   dim3 gridDim(ceil_div(N, BLOCK_SIZE));
  dim3 gridDim(1);

  // Warm-up iterations
  for (int i = 0; i < WARM_UP_ITERS; ++i) {
    test_kernel<<<gridDim, blockDim>>>(d_max_idx, d_max_val, d_input, N);
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
    test_kernel<<<gridDim, blockDim>>>(d_max_idx, d_max_val, d_input, N);
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
  CUDA_CHECK(cudaMemcpy(&h_max_idx_gpu, d_max_idx, sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_max_val_gpu, d_max_val, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU reference calculation
  argmax_cpu(&h_max_idx, &h_max_val, h_input);

  // Verify results
  bool verification_passed = (h_max_idx == h_max_idx_gpu) &&
                             verify_results(&h_max_val, &h_max_val_gpu, 1);

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
  if (d_max_idx)
    CUDA_CHECK(cudaFree(d_max_idx));
  if (d_max_val)
    CUDA_CHECK(cudaFree(d_max_val));
  if (d_input)
    CUDA_CHECK(cudaFree(d_input));
}