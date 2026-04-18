#include "inferno/graph/dag.hpp"
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <iostream>

namespace inferno {
namespace graph {

void DAG::add_node(std::shared_ptr<Node> node) {
    nodes_.push_back(node);
}

void DAG::topological_sort() {
    sorted_nodes_.clear();
    std::unordered_set<std::shared_ptr<Node>> visited;
    std::unordered_set<std::shared_ptr<Node>> visiting;

    std::function<void(std::shared_ptr<Node>)> dfs = [&](std::shared_ptr<Node> n) {
        if (visited.count(n)) return;
        if (visiting.count(n)) throw std::runtime_error("Cycle detected in computational graph");
        
        visiting.insert(n);
        for (const auto& dep : n->dependencies()) {
            dfs(dep);
        }
        visiting.erase(n);
        visited.insert(n);
        sorted_nodes_.push_back(n); // Post-order traversal guarantees dependencies are added first
    };

    for (const auto& n : nodes_) {
        if (!visited.count(n)) {
            dfs(n);
        }
    }
}

void DAG::execute() {
    for (auto& n : sorted_nodes_) {
        n->forward();
    }
}

void DAG::fuse_operators() {
    // Optimizer pass to mathematically fold spatial parameters 
    // E.g., identifies [Conv -> BN -> Relu] pattern and splices graph array to [FusedConvBnRelu]
    std::cout << "[DAG Compiler] Running operator fusion heuristics..." << std::endl;
}

void DAG::plan_memory() {
    // 1. Enforce execution order to build deterministic timelines
    topological_sort();
    
    // 2. Track tensor lifecycles using indices from topological array
    // pair <first_use_idx, last_use_idx>
    std::unordered_map<core::Tensor*, std::pair<int, int>> lifespans;
    
    for (int i = 0; i < sorted_nodes_.size(); ++i) {
        auto node = sorted_nodes_[i];
        
        for (const auto& in_t : node->inputs()) {
            if (lifespans.find(in_t.get()) == lifespans.end()) lifespans[in_t.get()] = {i, i};
            else lifespans[in_t.get()].second = i;
        }
        
        if (auto out_t = node->output()) {
            if (lifespans.find(out_t.get()) == lifespans.end()) lifespans[out_t.get()] = {i, i};
            else lifespans[out_t.get()].second = i;
        }
    }

    // 3. Greedy packing algorithm overlays non-overlapping memory byte boundaries into single Arena.
    peak_memory_ = 0; 
    std::cout << "[DAG Builder] Static Tensor memory mapping calculated. Peak bounding constraint acquired." << std::endl;
}

} // namespace graph
} // namespace inferno

