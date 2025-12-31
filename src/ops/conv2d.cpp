#include "inferno/ops/conv2d.hpp"
#include <stdexcept>

namespace inferno {
namespace ops {

void conv2d_naive(const core::Tensor& input, const core::Tensor& weight, core::Tensor& output) {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    auto out_shape = output.shape();

    if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
        throw std::invalid_argument("Conv2D requires 4D tensors");
    }

    size_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    size_t F = w_shape[0], KH = w_shape[2], KW = w_shape[3];
    size_t OH = out_shape[2], OW = out_shape[3];

    if (C != w_shape[1] || F != out_shape[1] || N != out_shape[0]) {
        throw std::invalid_argument("Dimension mismatch in Conv2D");
    }

    const float* in_data = input.raw_data();
    const float* w_data = weight.raw_data();
    float* out_data = output.raw_data();

    for(size_t i = 0; i < output.size(); ++i) out_data[i] = 0.0f;

    for (size_t n = 0; n < N; ++n) {
        for (size_t f = 0; f < F; ++f) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ ow) {
                    float sum = 0.0f;
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                size_t in_h = oh + kh;
                                size_t in_w = ow + kw;
                                
                                size_t in_idx = ((n*C + c)*H + in_h)*W + in_w;
                                size_t w_idx = ((f*C + c)*KH + kh)*KW + kw;
                                
                                sum += in_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }
                    size_t out_idx = ((n*F + f)*OH + oh)*OW + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
}

} // namespace ops
} // namespace inferno
