#pragma once
#include "inferno/core/tensor.hpp"

namespace inferno {
namespace ops {

// C = A * B
// A is [M x K], B is [K x N], C is [M x N]
void gemm_naive(const core::Tensor& A, const core::Tensor& B, core::Tensor& C);

// Tiled GEMM for L1/L2 Cache locality
void gemm_tiled(const core::Tensor& A, const core::Tensor& B, core::Tensor& C, size_t block_size = 32);

// AVX2 optimized GEMM
void gemm_avx2(const core::Tensor& A, const core::Tensor& B, core::Tensor& C);

} // namespace ops
} // namespace inferno
