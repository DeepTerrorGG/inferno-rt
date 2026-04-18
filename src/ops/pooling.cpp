#include "inferno/ops/pooling.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace inferno {
namespace ops {

void maxpool2d_naive(const core::Tensor& input, core::Tensor& output, size_t pool_size, size_t stride) {
    auto in_shape = input.shape();
    auto out_shape = output.shape();

    if (in_shape.size() != 4 || out_shape.size() != 4) {
        throw std::invalid_argument("MaxPooling requires 4D tensors");
    }

    size_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    size_t OH = out_shape[2], OW = out_shape[3];

    if (N != out_shape[0] || C != out_shape[1]) {
        throw std::invalid_argument("Dimension mismatch in MaxPooling");
    }

    const float* in_data = input.raw_data();
    float* out_data = output.raw_data();

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float max_val = std::numeric_limits<float>::lowest();
                    for (size_t ph = 0; ph < pool_size; ++ph) {
                        for (size_t pw = 0; pw < pool_size; ++pw) {
                            size_t h = oh * stride + ph;
                            size_t w = ow * stride + pw;
                            if (h < H && w < W) {
                                size_t in_idx = ((n*C + c)*H + h)*W + w;
                                max_val = std::max(max_val, in_data[in_idx]);
                            }
                        }
                    }
                    size_t out_idx = ((n*C + c)*OH + oh)*OW + ow;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }
}

} // namespace ops
} // namespace inferno

