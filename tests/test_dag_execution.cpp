#include <gtest/gtest.h>
#include "inferno/graph/dag.hpp"
#include "inferno/graph/ops_nodes.hpp"

using namespace inferno::core;
using namespace inferno::graph;

TEST(DAGTest, TopologicalSortMLP) {
    DAG dag;

    // Allocate Tensors
    auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 64});
    auto weight1 = std::make_shared<Tensor>(std::vector<size_t>{64, 32});
    auto hidden1 = std::make_shared<Tensor>(std::vector<size_t>{1, 32});
    
    // Fill specific inputs to test execution
    input->at({0, 0}) = 1.0f;
    weight1->at({0, 0}) = 2.0f;

    // Create Nodes
    auto fc1 = std::make_shared<GemmNode>("fc1");
    fc1->add_input(input);
    fc1->add_input(weight1);
    fc1->set_output(hidden1);

    auto relu1 = std::make_shared<ReluNode>("relu1");
    relu1->add_input(hidden1);
    relu1->add_dependency(fc1);

    // Register out-of-order to test topological sort engine
    dag.add_node(relu1); 
    dag.add_node(fc1);

    // Sort
    dag.topological_sort();
    
    auto sorted = dag.sorted_nodes();
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0]->name(), "fc1");
    EXPECT_EQ(sorted[1]->name(), "relu1");

    // Execute the forward pass through the graph
    dag.execute();

    // The single non-zero element from standard MatMul should be 1.0 * 2.0 = 2.0
    // ReLU simply passes it through since it's > 0
    EXPECT_FLOAT_EQ(hidden1->at({0, 0}), 2.0f);
}

