#pragma once

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>

#define BLOCK_SIZE 256
constexpr inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;

bool verify_results(const float *host_ref, const float *gpu_ref, int n,
                    float tolerance = 1e-8f) {
  for (int i = 0; i < n; ++i) {
    if (std::abs(host_ref[i] - gpu_ref[i]) > tolerance) {
      std::cerr << "Verification failed at index " << i << ":\n"
                << "  host: " << std::setprecision(10) << host_ref[i] << "\n"
                << "  gpu : " << std::setprecision(10) << gpu_ref[i] << "\n"
                << "  diff: " << std::abs(host_ref[i] - gpu_ref[i]) << " > "
                << tolerance << "\n";
      return false;
    }
  }
  // std::cout << "Verification passed (" << n << " elements).\n";
  return true;
}

typedef struct BenchmarkResults {
  std::string config_name;
  size_t size;
  double memory_usage_mb;
  dim3 block_dim;
  dim3 grid_dim;
  int warm_up_iters;
  int profile_iters;
  double avg_kernel_time_ms;
  double throughput_gbs;
  double compute_perf_gflops;
  bool verification_passed;
} BenchmarkResults;

void print_benchmark_results(const BenchmarkResults &results) {
  std::cout << "\n";
  std::cout << "=============== " << results.config_name
            << " ===============\n";
  std::cout << std::setw(22) << std::left << "Vector size:" << results.size
            << " elements\n";
  std::cout << std::setw(22) << std::left << "Data type:"
            << "float\n";
  std::cout << std::setw(22) << std::left << "Memory usage:" << std::fixed
            << std::setprecision(2) << results.memory_usage_mb << " MB\n";
  std::cout << std::setw(22) << std::left
            << "Block size:" << results.block_dim.x << " threads\n";
  std::cout << std::setw(22) << std::left << "Grid size:" << results.grid_dim.x
            << " blocks\n";
  std::cout << std::setw(22) << std::left
            << "Warm-up iters:" << results.warm_up_iters << "\n";
  std::cout << std::setw(22) << std::left
            << "Profile iters:" << results.profile_iters << "\n";
  std::cout << std::setw(22) << std::left << "Avg kernel time:" << std::fixed
            << std::setprecision(6) << results.avg_kernel_time_ms << " ms\n";
  std::cout << std::setw(22) << std::left << "Throughput:" << std::fixed
            << std::setprecision(2) << results.throughput_gbs << " GB/s\n";
  std::cout << std::setw(22) << std::left << "Compute perf:" << std::fixed
            << std::setprecision(2) << results.compute_perf_gflops
            << " GFLOP/s\n";
  std::cout << std::setw(22) << std::left << "Verification:"
            << (results.verification_passed ? "Passed" : "Failed") << "\n";
  std::cout << "================================================\n";
}