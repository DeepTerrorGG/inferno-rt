#pragma once
#include "inferno/graph/node.hpp"

namespace inferno {
namespace graph {

class FusedConvBnReluNode : public Node {
public:
    FusedConvBnReluNode(std::string name) : Node(std::move(name)) {}

    void forward() override {
        // High-level implementation:
        // By mathematically folding Batch Normalization variables (mean, var, beta, gamma)
        // into the Convolution weights and bias offline, the execution phase only has to 
        // run ONE loop traversal: Conv2D(with adjusted weights) -> BiasAdd -> ReLU.
        
        if (inputs_.empty() || !output_) return;

        // In production, we'd fire an optimized SIMD kernel here:
        // ops::fused_conv2d_bn_relu(input, fused_weights, fused_bias, output);

        // For this architectural backbone, we pass-through to denote successful graph traversal
        output_ = inputs_[0];
    }
};

} // namespace graph
} // namespace inferno
