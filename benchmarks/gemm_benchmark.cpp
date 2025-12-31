#include <benchmark/benchmark.h>
#include "inferno/ops/gemm.hpp"
#include <random>

using namespace inferno::core;
using namespace inferno::ops;

static void fill_random(Tensor& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    float* data = t.raw_data();
    for (size_t i = 0; i < t.size(); ++i) {
        data[i] = static_cast<float>(dis(gen));
    }
}

static void BM_GEMM_Naive(benchmark::State& state) {
    size_t size = state.range(0);
    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C({size, size});
    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        gemm_naive(A, B, C);
        benchmark::DoNotOptimize(C.raw_data());
    }
}

static void BM_GEMM_Tiled(benchmark::State& state) {
    size_t size = state.range(0);
    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C({size, size});
    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        gemm_tiled(A, B, C, 32);
        benchmark::DoNotOptimize(C.raw_data());
    }
}

static void BM_GEMM_AVX2(benchmark::State& state) {
    size_t size = state.range(0);
    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C({size, size});
    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        gemm_avx2(A, B, C);
        benchmark::DoNotOptimize(C.raw_data());
    }
}

#ifdef INFERNO_USE_CUDA
#include "inferno/backends/cuda_backend.cuh"

static void BM_GEMM_CUDA(benchmark::State& state) {
    size_t size = state.range(0);
    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C({size, size});
    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        inferno::backends::cuda_gemm(A, B, C);
        benchmark::DoNotOptimize(C.raw_data());
    }
}
BENCHMARK(BM_GEMM_CUDA)->RangeMultiplier(2)->Range(64, 512);
#endif

BENCHMARK(BM_GEMM_Naive)->RangeMultiplier(2)->Range(64, 512);
BENCHMARK(BM_GEMM_Tiled)->RangeMultiplier(2)->Range(64, 512);
BENCHMARK(BM_GEMM_AVX2)->RangeMultiplier(2)->Range(64, 512);

BENCHMARK_MAIN();
