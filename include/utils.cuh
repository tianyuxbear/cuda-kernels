#pragma once

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>

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

void fill_array_with_random_floats(float *base, size_t len,
                                   float min_val = 0.0f, float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (size_t i = 0; i < len; ++i) {
    base[i] = dis(gen);
  }
}

std::vector<int> random_sample_ints(int M, int N) {
  if (N > M) {
    throw std::invalid_argument("N cannot be greater than M");
  }
  if (N < 0 || M < 0) {
    throw std::invalid_argument("N and M must be non-negative");
  }

  std::vector<int> result;
  result.reserve(N);

  // 方法1: 当N较小时，使用拒绝采样
  if (N <= M / 2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, M - 1);
    std::unordered_set<int> selected;

    while (selected.size() < static_cast<size_t>(N)) {
      selected.insert(dis(gen));
    }

    result.assign(selected.begin(), selected.end());
  }
  // 方法2: 当N较大时，使用洗牌算法
  else {
    std::vector<int> pool(M);
    for (int i = 0; i < M; ++i) {
      pool[i] = i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pool.begin(), pool.end(), gen);

    result.assign(pool.begin(), pool.begin() + N);
  }

  return result;
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