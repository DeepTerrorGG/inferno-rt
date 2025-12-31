#pragma once
#include "inferno/core/tensor.hpp"

namespace inferno {
namespace ops {

// Naive MaxPooling2D
// Input: [N, C, H, W]
// Output: [N, C, OH, OW]
void maxpool2d_naive(const core::Tensor& input, core::Tensor& output, size_t pool_size, size_t stride);

} // namespace ops
} // namespace inferno
