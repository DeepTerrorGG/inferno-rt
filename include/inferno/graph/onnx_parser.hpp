#pragma once
#include "inferno/graph/dag.hpp"
#include <string>

namespace inferno {
namespace graph {

// Parses an ONNX model file and instantiates the computational graph
DAG parse_onnx(const std::string& filepath);

} // namespace graph
} // namespace inferno
