#pragma once
#include "inferno/graph/node.hpp"
#include "inferno/ops/gemm.hpp"
#include "inferno/ops/activations.hpp"

namespace inferno {
namespace graph {

class GemmNode : public Node {
public:
    GemmNode(std::string name) : Node(std::move(name)) {}
    void forward() override {
        // C = A * B
        if (inputs_.size() != 2 || !output_) return;
        ops::gemm_avx2(*inputs_[0], *inputs_[1], *output_);
    }
};

class ReluNode : public Node {
public:
    ReluNode(std::string name) : Node(std::move(name)) {}
    void forward() override {
        // A -> A (in-place)
        if (inputs_.size() != 1) return;
        ops::relu(*inputs_[0]);
        // Output tensor points to same underlying memory (in-place pass-through)
        output_ = inputs_[0]; 
    }
};

} // namespace graph
} // namespace inferno
