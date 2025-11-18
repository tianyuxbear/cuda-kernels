#include "check.cuh"

#include <stdio.h>

// 将字节转换为可读格式
void print_memory_size(size_t bytes) {
    if (bytes < 1024) {
        printf("%zu B", bytes);
    } else if (bytes < 1024 * 1024) {
        printf("%.2f KB", bytes / 1024.0);
    } else if (bytes < 1024 * 1024 * 1024) {
        printf("%.2f MB", bytes / (1024.0 * 1024.0));
    } else {
        printf("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}

// 打印设备详细信息
void print_device_properties(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("\n");
    printf("========================================\n");
    printf("Device %d: %s\n", device_id, prop.name);
    printf("========================================\n\n");

    // 基本信息
    printf("--- Basic Information ---\n");
    printf("  Compute Capability:           %d.%d\n", prop.major, prop.minor);
    printf("  Clock Rate:                   %.2f GHz\n", prop.clockRate / 1e6);
    printf("  Device Copy Overlap:          %s\n",
           prop.deviceOverlap ? "Enabled" : "Disabled");
    printf("  Kernel Execution Timeout:     %s\n",
           prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

    // 内存信息
    printf("\n--- Memory Information ---\n");
    printf("  Total Global Memory:          ");
    print_memory_size(prop.totalGlobalMem);
    printf("\n");
    printf("  Total Constant Memory:        ");
    print_memory_size(prop.totalConstMem);
    printf("\n");
    printf("  Shared Memory Per Block:      ");
    print_memory_size(prop.sharedMemPerBlock);
    printf("\n");
    printf("  L2 Cache Size:                ");
    print_memory_size(prop.l2CacheSize);
    printf("\n");
    printf("  Memory Clock Rate:            %.2f GHz\n",
           prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
    /*
      - Memory Clock Rate: 内存时钟频率 (Hz)
      - Memory Bus Width: 内存总线宽度 (bits)
      - × 2: DDR 效应（每时钟周期传输2次）
      - ÷ 8: 将 bits 转换为 bytes
      - ÷ 1e9: 转换为 GB/s
    */
    printf("  Peak Memory Bandwidth:        %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6);

    // 多处理器信息
    printf("\n--- Multiprocessor Information ---\n");
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Max Threads Per Multiprocessor: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Max Threads Per Block:        %d\n", prop.maxThreadsPerBlock);
    printf("  Max Thread Dimensions:        (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions:          (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size:                    %d threads\n", prop.warpSize);
    printf("  Registers Per Block:          %d\n", prop.regsPerBlock);
    printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);

    // 并发和异步特性
    printf("\n--- Concurrency Features ---\n");
    printf("  Concurrent Kernels:           %s\n",
           prop.concurrentKernels ? "Yes" : "No");
    printf("  Async Engine Count:           %d\n", prop.asyncEngineCount);
    printf("  Concurrent Copy and Execution: %s\n",
           (prop.asyncEngineCount > 0) ? "Yes" : "No");
    printf("  Stream Priorities Supported:  %s\n",
           prop.streamPrioritiesSupported ? "Yes" : "No");

    // 统一内存
    printf("\n--- Unified Memory ---\n");
    printf("  Managed Memory:               %s\n",
           prop.managedMemory ? "Supported" : "Not Supported");
    printf("  Concurrent Managed Access:    %s\n",
           prop.concurrentManagedAccess ? "Yes" : "No");
    printf("  Page Aligned Memory Required: %s\n",
           prop.pageableMemoryAccess ? "No" : "Yes");

    // 其他特性
    printf("\n--- Other Features ---\n");
    printf("  ECC Enabled:                  %s\n",
           prop.ECCEnabled ? "Yes" : "No");
    printf("  TCC Driver Mode:              %s\n", prop.tccDriver ? "Yes" : "No");
    printf("  Unified Addressing:           %s\n",
           prop.unifiedAddressing ? "Yes" : "No");
    printf("  Compute Mode:                 ");
    switch (prop.computeMode) {
    case cudaComputeModeDefault:
        printf("Default (multiple threads can use)\n");
        break;
    case cudaComputeModeExclusive:
        printf("Exclusive (only one thread can use)\n");
        break;
    case cudaComputeModeProhibited:
        printf("Prohibited (no threads can use)\n");
        break;
    case cudaComputeModeExclusiveProcess:
        printf("Exclusive Process\n");
        break;
    default:
        printf("Unknown\n");
    }

    // 计算最大理论性能
    printf("\n--- Performance Estimates ---\n");
    int cores_per_sm = 0;
    // 根据计算能力估算每个SM的CUDA核心数
    switch (prop.major) {
    case 2: // Fermi
        cores_per_sm = (prop.minor == 1) ? 48 : 32;
        break;
    case 3: // Kepler
        cores_per_sm = 192;
        break;
    case 5: // Maxwell
        cores_per_sm = 128;
        break;
    case 6: // Pascal
        cores_per_sm = (prop.minor == 0) ? 64 : 128;
        break;
    case 7: // Volta, Turing
        cores_per_sm = 64;
        break;
    case 8: // Ampere
        cores_per_sm = (prop.minor == 0) ? 64 : 128;
        break;
    case 9: // Hopper
        cores_per_sm = 128;
        break;
    default:
        cores_per_sm = 0;
    }

    if (cores_per_sm > 0) {
        int total_cores = cores_per_sm * prop.multiProcessorCount;
        printf("  CUDA Cores:                   %d (%d cores/SM * %d SMs)\n",
               total_cores, cores_per_sm, prop.multiProcessorCount);

        // 修正：先转换为 double，避免整数溢出
        // clockRate 单位是 kHz，需要除以 1e6 转换为 GHz
        // 每个 CUDA 核心每时钟周期可以执行 2 次浮点运算（FMA: 乘加融合）
        double clock_ghz = prop.clockRate / 1e6;                    // kHz -> GHz
        double peak_gflops = (double)total_cores * clock_ghz * 2.0; // * 2 因为 FMA

        printf("  Peak GFLOPS (FP32):           %.2f GFLOPS (single precision)\n",
               peak_gflops);
    }

    printf("\n");
}

// 打印当前内存使用情况
void print_memory_usage(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    printf("--- Current Memory Usage (Device %d) ---\n", device_id);
    printf("  Total Memory:  ");
    print_memory_size(total_mem);
    printf("\n");
    printf("  Free Memory:   ");
    print_memory_size(free_mem);
    printf("\n");
    printf("  Used Memory:   ");
    print_memory_size(total_mem - free_mem);
    printf(" (%.1f%%)\n", (total_mem - free_mem) * 100.0 / total_mem);
    printf("\n");
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║            CUDA Device Query - System Information             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    // 获取CUDA设备数量
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed! Error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("\nDetected %d CUDA capable device(s)\n", device_count);

    if (device_count == 0) {
        printf("\nNo CUDA capable devices found!\n");
        return 1;
    }

    // 获取CUDA驱动和运行时版本
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    printf("\nCUDA Driver Version:   %d.%d\n", driver_version / 1000,
           (driver_version % 100) / 10);
    printf("CUDA Runtime Version:  %d.%d\n", runtime_version / 1000,
           (runtime_version % 100) / 10);

    // 遍历所有设备
    for (int i = 0; i < device_count; i++) {
        print_device_properties(i);
        print_memory_usage(i);
    }

    // 打印当前活动设备
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    printf("Current active device: Device %d\n", current_device);

    printf(
        "\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                      Query Complete                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}