#pragma once
#include "inferno/core/tensor.hpp"

#ifdef INFERNO_USE_CUDA

namespace inferno {
namespace backends {

// Device Memory Allocator wrapper
void* cuda_allocate(size_t size);
void cuda_free(void* ptr);

// Host to Device (H2D)
void cuda_memcpy_h2d(void* dst, const void* src, size_t size);

// Device to Host (D2H)
void cuda_memcpy_d2h(void* dst, const void* src, size_t size);

// Custom CUDA kernels for ops
void cuda_gemm(const core::Tensor& A, const core::Tensor& B, core::Tensor& C);
void cuda_conv2d(const core::Tensor& input, const core::Tensor& weight, core::Tensor& output);

} // namespace backends
} // namespace inferno

#endif 
