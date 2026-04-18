#include <gtest/gtest.h>
#include "inferno/graph/dag.hpp"

using namespace inferno::graph;

TEST(FusionTest, OptimizerPassStub) {
    DAG dag;
    // Calling fuse_operators directly verifies compilation works smoothly
    EXPECT_NO_THROW(dag.fuse_operators());
}

