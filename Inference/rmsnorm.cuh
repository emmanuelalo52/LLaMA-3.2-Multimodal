#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
typedef __half floatX; // fp16 

namespace kernels{
    template<typename T,int BLOCK_SIZE, bool OUTPUT_RMS = false>
    __global__ void rmsnorm_kernel_vectorized(T* __restrict__ out, float* __restrict__ rms_out, const T* __restrict__ input, const T* __restrict__ weight, int N, int C, float eps){
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        int tx = blockIdx.x;
        if(tx>=N)return;

        const T* x = input + tx * C;
        T* o = out + tx * C;

        extern __shared__ float shared[];
        float* s_sum_sq = shared;

        // Vectorized path for fp16 
        constexpr int VEC_SIZE = 8;
        int C_vec = C/VEC_SIZE;
        float thread_sum_sq = 0.0f;

        const float4* x_vec = reinterpret_cast<const float4*>(x);

        for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE){
            float4 v = x_vec[i];
            __half2* h2_v = reinterpret_cast<__half2*>(&v);
            #pragma unroll
            for (int j = 0; j <4; j++){
                float2 f = __half22float2(h2_v[j]);
                thread_sum_sq += f.x *f.x + f.y * f.y;
            }
        }
        int remainder_start = C_vec *VEC_SIZE;
        for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE){
            float val = (float)x[i];
            thread_sum_sq += val * val;
        }
        float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int num_warps = BLOCK_SIZE/32;

        if(lane_id == 0){
            s_sum_sq[warp_id] = warp_sum;
        }
        block.sync();
        
        if(warp_id == 0){
            float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
            float block_sum = cg::reduce(warp,val,cg::plus<float>());

            if(lane_id == 0){
                float rms_inv = rsqrtf(block_sum / (float)C + eps);
                s_sum_sq[0] = rms_inv;
                if constexpr(OUTPUT_RMS){
                    rms_out[tx] = 1.0f/rms_inv;
                }
            }
        }
        block.sync(); // Wait for warp 0 to finish the calculation
        float rms_inv = s_sum_sq[0];
        
        // Normalization
        float4* o_vec = reinterpret_cast<float4*>(o);
        const float4* w_vec = reinterpret_cast<const float4*>(weight);

        for(int i = threadIdx.x; i < C_vec; i+=BLOCK_SIZE){
            float4 xv = x_vec[i];
            float4 wv = w_vec[i];

            __half2* h2_x = reinterpret_cast<__half2*>(&xv);
            __half2* h2_w = reinterpret_cast<__half2*>(&wv);

            float4 ov;
            __half2* h2_o = reinterpret_cast<__half2*>(&ov);

            #pragma unroll
            for(int j = 0; j < 4; j++){
                float2 xf = __half22float2(h2_x[j]);
                float2 wf = __half22float2(h2_w[j]);
                float2 of = {xf.x * rms_inv * wf.x, xf.y * rms_inv * wf.y};
                h2_o[j] = __float22half2_rn(of);
            }
            o_vec[i] = ov;
        }
        for(int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE){
            o[i] = (T)((float)x[i] * rms_inv * (float)weight[i]);
        }
    }
    template<typename T, int BLOCK_SIZE>
    __global__ void rmsnorm_backward_kernel(T* d_inp, float* d_weight, const T* grad, 
                                            const T* inp, const T* weight, const float* rms, 
                                            int N, int C) {
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        int idx = blockIdx.x;
        if (idx >= N) return;

        const T* x = inp + idx * C;
        const T* g = grad + idx * C;
        T* dx = d_inp + idx * C;
        
        float r_inv = 1.0f / rms[idx];

        extern __shared__ float shared[];
        float* s_reduce = shared;

        float thread_dot = 0.0f;
        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            float g_val = (float)g[i];
            float x_val = (float)x[i];
            float w_val = (float)weight[i];
            thread_dot += g_val * w_val * x_val;
            
            atomicAdd(&d_weight[i], g_val * x_val * r_inv);
        }

        float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) s_reduce[warp_id] = warp_dot;
        block.sync();

        if (warp_id == 0) {
            float val = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
            float block_dot = cg::reduce(warp, val, cg::plus<float>());
            if (lane_id == 0) s_reduce[0] = block_dot;
        }
        block.sync();

        float correction = s_reduce[0] / (C * (1.0f / r_inv) * (1.0f / r_inv));

        for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
            dx[i] = (T)(r_inv * ((float)g[i] * (float)weight[i] - (float)x[i] * correction));
        }
    }
}


