#pragma once
#include "inferno/core/tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace inferno {
namespace graph {

class Node {
public:
    Node(std::string name) : name_(std::move(name)) {}
    virtual ~Node() = default;

    virtual void forward() = 0;

    void add_input(std::shared_ptr<core::Tensor> t) { inputs_.push_back(t); }
    void set_output(std::shared_ptr<core::Tensor> t) { output_ = t; }

    const std::vector<std::shared_ptr<core::Tensor>>& inputs() const { return inputs_; }
    std::shared_ptr<core::Tensor> output() const { return output_; }

    const std::string& name() const { return name_; }

    void add_dependency(std::shared_ptr<Node> node) { dependencies_.push_back(node); }
    const std::vector<std::shared_ptr<Node>>& dependencies() const { return dependencies_; }

protected:
    std::string name_;
    std::vector<std::shared_ptr<core::Tensor>> inputs_;
    std::shared_ptr<core::Tensor> output_;
    std::vector<std::shared_ptr<Node>> dependencies_;
};

} // namespace graph
} // namespace inferno
