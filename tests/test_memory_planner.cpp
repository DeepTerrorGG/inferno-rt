#include <gtest/gtest.h>
#include "inferno/graph/dag.hpp"
#include "inferno/graph/ops_nodes.hpp"

using namespace inferno::graph;

TEST(MemoryPlannerTest, CalculateLifespans) {
    DAG dag;
    auto node = std::make_shared<ReluNode>("fake_relu");
    dag.add_node(node);
    
    EXPECT_NO_THROW(dag.plan_memory());
    EXPECT_EQ(dag.peak_memory_footprint(), 0); 
}

