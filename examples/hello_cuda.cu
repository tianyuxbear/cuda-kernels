#include "check.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数 - 在GPU上运行
__global__ void hello_from_gpu() {
  // 获取线程的唯一标识
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello from GPU! Thread %d in Block %d (Global ID: %d)\n", threadIdx.x,
         blockIdx.x, tid);
}

// 主机函数
void hello_from_cpu() { printf("Hello from CPU!\n"); }

int main() {
  printf("=== CUDA Hello World Example ===\n\n");

  // 1. CPU端打印
  hello_from_cpu();
  printf("\n");

  // 2. 检查CUDA设备
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  printf("Found %d CUDA device(s)\n\n", device_count);

  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices found!\n");
    return 1;
  }

  // 3. 启动CUDA核函数
  // <<<blocks, threads_per_block>>>
  printf("Launching kernel with 4 blocks, 4 threads per block...\n\n");
  hello_from_gpu<<<4, 4>>>();

  // 4. 等待GPU完成
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\n=== Program completed successfully ===\n");

  return 0;
}