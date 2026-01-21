add_executable(linear_fp32 linear_fp32.cu)
add_executable(linear_fp16 linear_fp16.cu)
add_executable(linear_bf16 linear_bf16.cu)
add_executable(test_wmma_layout test_wmma_layout.cu)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(linear_bf16
    PRIVATE
    CUDA::cublas
    CUDA::cublasLt
)
