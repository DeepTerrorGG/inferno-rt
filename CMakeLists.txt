cmake_minimum_required(VERSION 3.20)
project(InfernoRT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# AVX
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Protobuf
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v25.1
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)

# Download ONNX schema
set(ONNX_PROTO_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto)
if (NOT EXISTS ${ONNX_PROTO_FILE})
  message(STATUS "Downloading onnx.proto from github...")
  file(DOWNLOAD "https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto" ${ONNX_PROTO_FILE} TIMEOUT 30)
endif()

# Protobuf generate
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
    COMMAND protobuf::protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I=${CMAKE_CURRENT_BINARY_DIR} ${ONNX_PROTO_FILE}
    DEPENDS ${ONNX_PROTO_FILE} protobuf::protoc
    COMMENT "Compiling ONNX Protobuf Schema"
)
add_library(inferno_onnx_proto ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)
target_link_libraries(inferno_onnx_proto PUBLIC protobuf::libprotobuf)
target_include_directories(inferno_onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Core Library
add_library(inferno_core
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/ops/activations.cpp
    src/ops/gemm.cpp
    src/ops/conv2d.cpp
    src/ops/pooling.cpp
)

option(USE_CUDA "Enable CUDA backend support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(INFERNO_USE_CUDA)
    target_sources(inferno_core PRIVATE src/backends/cuda_backend.cu)
    target_link_libraries(inferno_core PUBLIC CUDA::cudart)
endif()

target_include_directories(inferno_core PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(inferno_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Graph Library
add_library(inferno_graph
    src/graph/dag.cpp
    src/graph/onnx_parser.cpp
)
target_include_directories(inferno_graph PUBLIC include)
target_link_libraries(inferno_graph PUBLIC inferno_core inferno_onnx_proto)

# Tests
enable_testing()
add_executable(test_tensor_strides tests/test_tensor_strides.cpp)
target_link_libraries(test_tensor_strides PRIVATE inferno_core GTest::gtest_main)

add_executable(test_gemm_accuracy tests/test_gemm_accuracy.cpp)
target_link_libraries(test_gemm_accuracy PRIVATE inferno_core GTest::gtest_main)

add_executable(test_dag_execution tests/test_dag_execution.cpp)
target_link_libraries(test_dag_execution PRIVATE inferno_graph GTest::gtest_main)

add_executable(test_onnx_parser tests/test_onnx_parser.cpp)
target_link_libraries(test_onnx_parser PRIVATE inferno_graph GTest::gtest_main)

add_executable(test_operator_fusion tests/test_operator_fusion.cpp)
target_link_libraries(test_operator_fusion PRIVATE inferno_graph GTest::gtest_main)

add_executable(test_memory_planner tests/test_memory_planner.cpp)
target_link_libraries(test_memory_planner PRIVATE inferno_graph GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor_strides)
gtest_discover_tests(test_gemm_accuracy)
gtest_discover_tests(test_dag_execution)
gtest_discover_tests(test_onnx_parser)
gtest_discover_tests(test_operator_fusion)
gtest_discover_tests(test_memory_planner)

# Benchmarks
add_executable(gemm_benchmark benchmarks/gemm_benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE inferno_core benchmark::benchmark benchmark::benchmark_main)

# Examples
add_executable(realtime_eval examples/realtime_eval.cpp)
target_link_libraries(realtime_eval PRIVATE inferno_core)

# --- Python Bindings Showcase (pybind11) ---
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(inferno_python src/python/bindings.cpp)
    set_target_properties(inferno_python PROPERTIES OUTPUT_NAME "inferno")
    target_link_libraries(inferno_python PRIVATE inferno_graph)
endif()
