#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdlib>

#include "inferno/core/tensor.hpp"
#include "inferno/graph/dag.hpp"
#include "inferno/graph/onnx_parser.hpp"

namespace py = pybind11;
using namespace inferno::core;
using namespace inferno::graph;

PYBIND11_MODULE(inferno, m) {
    m.doc() = "Inferno-RT: Bare-metal neural network inference engine";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>>())
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("fill_random", [](Tensor& t) {
            float* data = t.raw_data();
            for (size_t i = 0; i < t.size(); ++i) {
                // Highly generic uniform fill for showcase
                data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        })
        .def("to_numpy", [](Tensor& t) {
            auto shape = t.shape();
            std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
            return py::array_t<float>(py_shape, t.raw_data(), py::cast(t));
        });

    py::class_<DAG>(m, "DAG")
        .def(py::init<>())
        .def("topological_sort", &DAG::topological_sort)
        .def("plan_memory", &DAG::plan_memory)
        .def("fuse_operators", &DAG::fuse_operators)
        .def("peak_memory_footprint", &DAG::peak_memory_footprint);

    m.def("parse_onnx", &parse_onnx, "Parses an ONNX model into a DAG architecture",
          py::arg("filepath"));
}

