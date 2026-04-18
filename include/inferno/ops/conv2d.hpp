#pragma once
#include "inferno/core/tensor.hpp"

namespace inferno {
namespace ops {

// Naive Conv2D without padding/strides support
// Input: [N, C, H, W]
// Weight: [F, C, KH, KW]
// Output: [N, F, OH, OW]
void conv2d_naive(const core::Tensor& input, const core::Tensor& weight, core::Tensor& output);

} // namespace ops
} // namespace inferno

