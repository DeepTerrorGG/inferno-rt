#include "inferno/ops/activations.hpp"
#include <algorithm>
#include <immintrin.h>

namespace inferno {
namespace ops {

void relu(core::Tensor& t) {
    float* data = t.raw_data();
    size_t size = t.size();
    
    size_t i = 0;
    __m256 zeros = _mm256_setzero_ps();
    
    // Process using AVX2 SIMD max operation
    for (; i + 8 <= size; i += 8) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_max_ps(vals, zeros);
        _mm256_storeu_ps(&data[i], vals);
    }
    
    // Tail
    for (; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

} // namespace ops
} // namespace inferno
