#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace kernels {
    template<typename T, int BLOCK_SIZE, bool OUTPUT_RMS = false>
    __global__ void rmsnorm_kernel_vectorized(T* __restrict__ out, float* __restrict__ rms_out, const T* __restrict__ input, const T* __restrict__ weight, int N, int C, float eps) {
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        int tx = blockIdx.x;
        if (tx >= N) return;

        const T* x = input + tx * C;
        T* o = out + tx * C;

        extern __shared__ float s_reduce[];

        // 1. Calculate Sum of Squares
        constexpr int VEC_SIZE = 4; // 4 halfs in a uint2 (64 bits)
        int C_vec = C / VEC_SIZE;
        float thread_sum_sq = 0.0f;

        const uint2* x_vec = reinterpret_cast<const uint2*>(x);

        for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
            uint2 v = x_vec[i];
            __half2* h2_v = reinterpret_cast<__half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 2; j++) { // Corrected: uint2 has 2 half2 elements
                float2 f = __half22float2(h2_v[j]);
                thread_sum_sq += f.x * f.x + f.y * f.y;
            }
        }
        
        // Handle remainder if C is not multiple of 4
        for (int i = C_vec * VEC_SIZE + threadIdx.x; i < C; i += BLOCK_SIZE) {
            float val = (float)x[i];
            thread_sum_sq += val * val;
        }

        // First reduce within warps
        float warp_sum_sq = cg::reduce(warp, thread_sum_sq, cg::plus<float>());
        
        // Then reduce across warps using shared memory
        if (warp.thread_rank() == 0) {
            s_reduce[threadIdx.x / 32] = warp_sum_sq;
        }
        block.sync();
        
        // Final reduction by first warp
        float block_sum_sq = 0.0f;
        if (threadIdx.x < (BLOCK_SIZE + 31) / 32) {
            block_sum_sq = s_reduce[threadIdx.x];
        }
        if (threadIdx.x < 32) {
            block_sum_sq = cg::reduce(warp, block_sum_sq, cg::plus<float>());
        }

        if (threadIdx.x == 0) {
            float rms_inv = rsqrtf(block_sum_sq / (float)C + eps);
            s_reduce[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[tx] = 1.0f / rms_inv;
            }
        }
        block.sync();
        float rms_inv = s_reduce[0];

        // 2. Normalization and Scaling
        uint2* o_vec = reinterpret_cast<uint2*>(o);
        const uint2* w_vec = reinterpret_cast<const uint2*>(weight);

        for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
            uint2 xv = x_vec[i];
            uint2 wv = w_vec[i];
            __half2* h2_x = reinterpret_cast<__half2*>(&xv);
            __half2* h2_w = reinterpret_cast<__half2*>(&wv);
            uint2 result;
            __half2* h2_res = reinterpret_cast<__half2*>(&result);

            #pragma unroll
            for (int j = 0; j < 2; j++) {
                float2 x_f2 = __half22float2(h2_x[j]);
                float2 w_f2 = __half22float2(h2_w[j]);
                float2 res_f2 = {x_f2.x * rms_inv * w_f2.x, x_f2.y * rms_inv * w_f2.y};
                h2_res[j] = __float22half2_rn(res_f2);
            }
            o_vec[i] = result;
        }
        
        for (int i = C_vec * VEC_SIZE + threadIdx.x; i < C; i += BLOCK_SIZE) {
            o[i] = (T)((float)x[i] * rms_inv * (float)weight[i]);
        }
    }

    template<typename T, int BLOCK_SIZE>
    __global__ void rmsnorm_backward_kernel(T* d_inp, float* d_weight, const T* grad, 
                                            const T* inp, const T* weight, const float* rms, 
                                            int N, int C) {
        int idx = blockIdx.x;
        if (idx >= N) return;

        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        const T* x = inp + idx * C;
        const T* g = grad + idx * C;
        T* dx = d_inp + idx * C;
        
        float r_inv = 1.0f / (rms[idx] + 1e-6f);
        extern __shared__ float s_reduce[];

        float thread_dot = 0.0f;
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float g_val = (float)g[i];
            float x_val = (float)x[i];
            float w_val = (float)weight[i];
            thread_dot += g_val * w_val * x_val;
            
            // Note: Parallel d_weight calculation is usually done in a separate kernel 
            // to avoid atomicAdd contention. For now, we fix the logic.
            atomicAdd(&d_weight[i], g_val * x_val * r_inv);
        }

        // Warp-level reduction first
        float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            s_reduce[threadIdx.x / 32] = warp_dot;
        }
        block.sync();
        
        // Final reduction by first warp
        float block_dot = 0.0f;
        if (threadIdx.x < (BLOCK_SIZE + 31) / 32) {
            block_dot = s_reduce[threadIdx.x];
        }
        if (threadIdx.x < 32) {
            block_dot = cg::reduce(warp, block_dot, cg::plus<float>());
        }
        
        if (threadIdx.x == 0) s_reduce[0] = block_dot;
        block.sync();

        float correction = s_reduce[0] / (C * rms[idx] * rms[idx]);

        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            dx[i] = (T)(r_inv * ((float)g[i] * (float)weight[i] - (float)x[i] * correction));
        }
    }
}