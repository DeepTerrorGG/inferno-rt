#include <gtest/gtest.h>
#include "inferno/graph/onnx_parser.hpp"

using namespace inferno::graph;

TEST(ONNXParserTest, EmptyFileThrowsRuntimeError) {
    // Proves the ONNX parser natively compiles and handles invalid files cleanly
    EXPECT_THROW(parse_onnx("non_existent_model.onnx"), std::runtime_error);
}

