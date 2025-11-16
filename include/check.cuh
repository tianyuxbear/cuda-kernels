#pragma once

#include <cuda_runtime.h> // for cudaError_t, cudaSuccess, cudaGetErrorString
#include <stdexcept>      // IWYU pragma: keep
#include <stdio.h>        // IWYU pragma: keep  for fprintf, stderr
#include <stdlib.h>       // for exit()

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);              \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define EXCEPTION_LOCATION_MSG                                                 \
  " from " << __func__ << " at " << __FILE__ << ":" << __LINE__ << "."

#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "[ERROR] " << message << std::endl                          \
                << "Assertion failed: " << #condition                          \
                << EXCEPTION_LOCATION_MSG << std::endl;                        \
      throw std::runtime_error("Assertion failed");                            \
    }                                                                          \
  } while (0)
