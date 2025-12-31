#include <gtest/gtest.h>
#include "inferno/ops/gemm.hpp"
#include <random>
#include <cmath>

using namespace inferno::core;
using namespace inferno::ops;

static void fill_random(Tensor& t) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    float* data = t.raw_data();
    for (size_t i = 0; i < t.size(); ++i) {
        data[i] = static_cast<float>(dis(gen));
    }
}

TEST(GemmTest, AccuracyCompare) {
    size_t M = 128, K = 128, N = 128;
    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_naive({M, N});
    Tensor C_tiled({M, N});
    Tensor C_avx2({M, N});

    fill_random(A);
    fill_random(B);

    gemm_naive(A, B, C_naive);
    gemm_tiled(A, B, C_tiled, 32);
    gemm_avx2(A, B, C_avx2);

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_naive.raw_data()[i], C_tiled.raw_data()[i], 1e-4);
        EXPECT_NEAR(C_naive.raw_data()[i], C_avx2.raw_data()[i], 1e-4);
    }
}
