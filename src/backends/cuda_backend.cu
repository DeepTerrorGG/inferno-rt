#ifdef INFERNO_USE_CUDA

#include "inferno/backends/cuda_backend.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace inferno {
namespace backends {

void* cuda_allocate(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

__global__ void gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void cuda_gemm(const core::Tensor& A, const core::Tensor& B, core::Tensor& C) {
    int M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    d_A = (float*)cuda_allocate(size_A);
    d_B = (float*)cuda_allocate(size_B);
    d_C = (float*)cuda_allocate(size_C);

    cuda_memcpy_h2d(d_A, A.raw_data(), size_A);
    cuda_memcpy_h2d(d_B, B.raw_data(), size_B);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    gemm_kernel<<<blocks, threads>>>(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_memcpy_d2h(C.raw_data(), d_C, size_C);

    cuda_free(d_A);
    cuda_free(d_B);
    cuda_free(d_C);
}

__global__ void conv2d_kernel(int N, int C, int H, int W, int F, int KH, int KW, int OH, int OW, const float* in, const float* w, float* out) {
    int f = blockIdx.z; 
    int oh = blockIdx.y * blockDim.y + threadIdx.y; 
    int ow = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (f < F && oh < OH && ow < OW) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int in_h = oh + kh;
                        int in_w = ow + kw;
                        int in_idx = ((n * C + c) * H + in_h) * W + in_w;
                        int w_idx = ((f * C + c) * KH + kh) * KW + kw;
                        sum += in[in_idx] * w[w_idx];
                    }
                }
            }
            int out_idx = ((n * F + f) * OH + oh) * OW + ow;
            out[out_idx] = sum; 
        }
    }
}

void cuda_conv2d(const core::Tensor& input, const core::Tensor& weight, core::Tensor& output) {
    size_t in_size = input.size() * sizeof(float);
    size_t w_size = weight.size() * sizeof(float);
    size_t out_size = output.size() * sizeof(float);

    float *d_in, *d_w, *d_out;
    d_in = (float*)cuda_allocate(in_size);
    d_w = (float*)cuda_allocate(w_size);
    d_out = (float*)cuda_allocate(out_size);

    cuda_memcpy_h2d(d_in, input.raw_data(), in_size);
    cuda_memcpy_h2d(d_w, weight.raw_data(), w_size);

    int N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    int F = weight.shape()[0], KH = weight.shape()[2], KW = weight.shape()[3];
    int OH = output.shape()[2], OW = output.shape()[3];

    dim3 threads(16, 16);
    dim3 blocks((OW + threads.x - 1) / threads.x, (OH + threads.y - 1) / threads.y, F);

    conv2d_kernel<<<blocks, threads>>>(N, C, H, W, F, KH, KW, OH, OW, d_in, d_w, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_memcpy_d2h(output.raw_data(), d_out, out_size);

    cuda_free(d_in);
    cuda_free(d_w);
    cuda_free(d_out);
}

} // namespace backends
} // namespace inferno

#endif

