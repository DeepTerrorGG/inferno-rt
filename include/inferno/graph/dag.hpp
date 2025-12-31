#pragma once
#include "inferno/graph/node.hpp"
#include <vector>
#include <memory>

namespace inferno {
namespace graph {

class DAG {
public:
    DAG() = default;

    void add_node(std::shared_ptr<Node> node);

    // Sorts the nodes topologically based on their dependencies
    void topological_sort();

    // Optimizer Pass: Operator Fusion
    void fuse_operators();

    // Builder Pass: Static Memory Planning
    void plan_memory();

    // Peak global tensor memory boundary
    size_t peak_memory_footprint() const { return peak_memory_; }

    // Executes the nodes in topologically sorted order
    void execute();

    const std::vector<std::shared_ptr<Node>>& sorted_nodes() const { return sorted_nodes_; }

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<std::shared_ptr<Node>> sorted_nodes_;
    size_t peak_memory_ = 0;
};

} // namespace graph
} // namespace inferno
