cmake_minimum_required(VERSION 3.18)

project(CUDA-Kernels LANGUAGES CXX CUDA)

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architecture (e.g., Ada: SM89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")          # Line info for profiling
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")     # Debug symbols
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")# High optimization

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add example executables (Hello World, device info, etc.)
add_subdirectory(examples)

# Add elementwise kernels (add, etc.)
add_subdirectory(src/elementwise)

# Add reduce kernels (max, etc.)
add_subdirectory(src/reduce)

# Add embedding kernels
add_subdirectory(src/embedding)

# Add normalize kernels
add_subdirectory(src/norm)

# Add matmul kernels
add_subdirectory(src/linear)
