#include "inferno/ops/gemm.hpp"
#include <stdexcept>
#include <immintrin.h>

namespace inferno {
namespace ops {

void check_gemm_shapes(const core::Tensor& A, const core::Tensor& B, const core::Tensor& C) {
    if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2) {
        throw std::invalid_argument("GEMM requires 2D tensors");
    }
    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument("A inner dimension must match B inner dimension");
    }
    if (C.shape()[0] != A.shape()[0] || C.shape()[1] != B.shape()[1]) {
        throw std::invalid_argument("C shape must match AxB outer dimensions");
    }
}

void gemm_naive(const core::Tensor& A, const core::Tensor& B, core::Tensor& C) {
    check_gemm_shapes(A, B, C);
    
    size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    const float* a_data = A.raw_data();
    const float* b_data = B.raw_data();
    float* c_data = C.raw_data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }
}

void gemm_tiled(const core::Tensor& A, const core::Tensor& B, core::Tensor& C, size_t block_size) {
    check_gemm_shapes(A, B, C);
    
    size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    const float* a_data = A.raw_data();
    const float* b_data = B.raw_data();
    float* c_data = C.raw_data();

    for(size_t i = 0; i < M * N; ++i) c_data[i] = 0.0f;

    for (size_t i0 = 0; i0 < M; i0 += block_size) {
        size_t imax = std::min(i0 + block_size, M);
        for (size_t j0 = 0; j0 < N; j0 += block_size) {
            size_t jmax = std::min(j0 + block_size, N);
            for (size_t k0 = 0; k0 < K; k0 += block_size) {
                size_t kmax = std::min(k0 + block_size, K);
                
                for (size_t i = i0; i < imax; ++i) {
                    for (size_t k = k0; k < kmax; ++k) {
                        float a_val = a_data[i * K + k];
                        for (size_t j = j0; j < jmax; ++j) {
                            c_data[i * N + j] += a_val * b_data[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_avx2(const core::Tensor& A, const core::Tensor& B, core::Tensor& C) {
    check_gemm_shapes(A, B, C);
    
    size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    const float* a_data = A.raw_data();
    const float* b_data = B.raw_data();
    float* c_data = C.raw_data();

    for(size_t i = 0; i < M * N; ++i) c_data[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_val = a_data[i * K + k];
            __m256 va = _mm256_set1_ps(a_val);
            size_t j = 0;
            
            for (; j + 8 <= N; j += 8) {
                __m256 vb = _mm256_loadu_ps(&b_data[k * N + j]);
                __m256 vc = _mm256_loadu_ps(&c_data[i * N + j]);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(&c_data[i * N + j], vc);
            }
            for (; j < N; ++j) {
                c_data[i * N + j] += a_val * b_data[k * N + j];
            }
        }
    }
}

} // namespace ops
} // namespace inferno
