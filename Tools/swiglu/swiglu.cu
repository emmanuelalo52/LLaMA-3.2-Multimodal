#include "swiglu.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// SiLU activation: x * sigmoid(x)
template<typename T>
__device__ __forceinline__ T silu(T x) {
    return x / (static_cast<T>(1.0) + expf(-x));
}

// Specialized SiLU for half precision
__device__ __forceinline__ __half silu(__half x) {
    return __hmul(x, hrcp(__hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

// Specialized SiLU for bfloat16
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ __nv_bfloat16 silu(__nv_bfloat16 x) {
    return __hmul(x, hrcp(__hadd(__float2bfloat16(1.0f), hexp(__hneg(x)))));
}
#endif

// SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__device__ __forceinline__ T silu_grad(T x, T grad_output) {
    T sigmoid_x = static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-x));
    return grad_output * sigmoid_x * (static_cast<T>(1.0) + x * (static_cast<T>(1.0) - sigmoid_x));
}

/**
 * Fused SwiGLU Forward Kernel
 * 
 * Computes: output = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
 * 
 * This kernel fuses:
 * 1. Gate projection: x @ W_gate + b_gate
 * 2. Up projection: x @ W_up + b_up
 * 3. SiLU activation on gate
 * 4. Element-wise multiplication
 * 
 * Memory layout:
 * - x: [batch_size, seq_len, hidden_size]
 * - W_gate, W_up: [hidden_size, intermediate_size]
 * - output: [batch_size, seq_len, intermediate_size]
 */
template<typename T>
__global__ void swiglu_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    T* __restrict__ output,
    T* __restrict__ gate_cache,  // Cache for backward pass
    T* __restrict__ up_cache,    // Cache for backward pass
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    // Grid-stride loop over output elements
    int total_elements = batch_size * seq_len * intermediate_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Compute batch, sequence, and feature indices
        int inter_idx = idx % intermediate_size;
        int seq_idx = (idx / intermediate_size) % seq_len;
        int batch_idx = idx / (seq_len * intermediate_size);
        
        // Offset to the input row
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        // Compute gate projection
        T gate_val = b_gate ? b_gate[inter_idx] : static_cast<T>(0.0);
        
        // Compute up projection
        T up_val = b_up ? b_up[inter_idx] : static_cast<T>(0.0);
        
        // Matrix multiplication for both projections
        #pragma unroll 4
        for (int h = 0; h < hidden_size; h++) {
            T x_val = x[x_offset + h];
            gate_val += x_val * w_gate[h * intermediate_size + inter_idx];
            up_val += x_val * w_up[h * intermediate_size + inter_idx];
        }
        
        // Apply SiLU to gate
        T gate_activated = silu(gate_val);
        
        // Element-wise multiplication
        T result = gate_activated * up_val;
        
        // Store output and cache for backward
        output[idx] = result;
        if (gate_cache) gate_cache[idx] = gate_val;
        if (up_cache) up_cache[idx] = up_val;
    }
}

/**
 * Optimized SwiGLU Forward Kernel using Shared Memory
 * 
 * This version uses shared memory to cache input features and weight tiles
 * for better memory coalescing and reduced global memory access.
 */
template<typename T, int TILE_SIZE = 32>
__global__ void swiglu_forward_optimized_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    T* __restrict__ output,
    T* __restrict__ gate_cache,
    T* __restrict__ up_cache,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    // Shared memory for input tile and weight tiles
    __shared__ T s_x[TILE_SIZE];
    __shared__ T s_w_gate[TILE_SIZE];
    __shared__ T s_w_up[TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int inter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || inter_idx >= intermediate_size) {
        return;
    }
    
    int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
    int out_idx = (batch_idx * seq_len + seq_idx) * intermediate_size + inter_idx;
    
    T gate_val = b_gate ? b_gate[inter_idx] : static_cast<T>(0.0);
    T up_val = b_up ? b_up[inter_idx] : static_cast<T>(0.0);
    
    // Process hidden dimension in tiles
    for (int tile = 0; tile < (hidden_size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int h_idx = tile * TILE_SIZE + threadIdx.x;
        
        // Load input tile cooperatively
        if (h_idx < hidden_size && threadIdx.x < TILE_SIZE) {
            s_x[threadIdx.x] = x[x_offset + h_idx];
        } else {
            s_x[threadIdx.x] = static_cast<T>(0.0);
        }
        
        // Load weight tiles
        if (h_idx < hidden_size && inter_idx < intermediate_size) {
            s_w_gate[threadIdx.x] = w_gate[h_idx * intermediate_size + inter_idx];
            s_w_up[threadIdx.x] = w_up[h_idx * intermediate_size + inter_idx];
        } else {
            s_w_gate[threadIdx.x] = static_cast<T>(0.0);
            s_w_up[threadIdx.x] = static_cast<T>(0.0);
        }
        
        __syncthreads();
        
        // Compute dot products
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            if (tile * TILE_SIZE + i < hidden_size) {
                gate_val += s_x[i] * s_w_gate[i];
                up_val += s_x[i] * s_w_up[i];
            }
        }
        
        __syncthreads();
    }
    
    // Apply SiLU and multiply
    T gate_activated = silu(gate_val);
    T result = gate_activated * up_val;
    
    // Store results
    output[out_idx] = result;
    if (gate_cache) gate_cache[out_idx] = gate_val;
    if (up_cache) up_cache[out_idx] = up_val;
}

/**
 * Fused SwiGLU Backward Kernel
 * 
 * Computes gradients for x, w_gate, and w_up given grad_output
 */
template<typename T>
__global__ void swiglu_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ gate_cache,
    const T* __restrict__ up_cache,
    T* __restrict__ grad_x,
    T* __restrict__ grad_w_gate,
    T* __restrict__ grad_w_up,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    int total_elements = batch_size * seq_len * intermediate_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int inter_idx = idx % intermediate_size;
        int seq_idx = (idx / intermediate_size) % seq_len;
        int batch_idx = idx / (seq_len * intermediate_size);
        
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        T grad_out = grad_output[idx];
        T gate_val = gate_cache[idx];
        T up_val = up_cache[idx];
        
        // Gradient through multiplication
        T gate_activated = silu(gate_val);
        T grad_gate_activated = grad_out * up_val;
        T grad_up = grad_out * gate_activated;
        
        // Gradient through SiLU
        T grad_gate = silu_grad(gate_val, grad_gate_activated);
        
        // Backprop to input and weights
        for (int h = 0; h < hidden_size; h++) {
            T x_val = x[x_offset + h];
            
            // Gradient w.r.t. input (accumulate from both paths)
            atomicAdd(&grad_x[x_offset + h], 
                     grad_gate * w_gate[h * intermediate_size + inter_idx] +
                     grad_up * w_up[h * intermediate_size + inter_idx]);
            
            // Gradient w.r.t. weights
            atomicAdd(&grad_w_gate[h * intermediate_size + inter_idx], grad_gate * x_val);
            atomicAdd(&grad_w_up[h * intermediate_size + inter_idx], grad_up * x_val);
        }
    }
}

/**
 * Fused SwiGLU + Down Projection Forward Kernel
 * 
 * Computes the entire feedforward layer in one kernel:
 * output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
 */
template<typename T>
__global__ void swiglu_down_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_up,
    const T* __restrict__ w_down,
    const T* __restrict__ b_gate,
    const T* __restrict__ b_up,
    const T* __restrict__ b_down,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    // Each thread computes one output element
    int total_elements = batch_size * seq_len * hidden_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int h_out = idx % hidden_size;
        int seq_idx = (idx / hidden_size) % seq_len;
        int batch_idx = idx / (seq_len * hidden_size);
        
        int x_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        
        T result = b_down ? b_down[h_out] : static_cast<T>(0.0);
        
        // Compute intermediate activations and project down
        for (int inter = 0; inter < intermediate_size; inter++) {
            // Compute gate and up projections for this intermediate dimension
            T gate_val = b_gate ? b_gate[inter] : static_cast<T>(0.0);
            T up_val = b_up ? b_up[inter] : static_cast<T>(0.0);
            
            #pragma unroll 4
            for (int h = 0; h < hidden_size; h++) {
                T x_val = x[x_offset + h];
                gate_val += x_val * w_gate[h * intermediate_size + inter];
                up_val += x_val * w_up[h * intermediate_size + inter];
            }
            
            // Apply SwiGLU
            T swiglu_out = silu(gate_val) * up_val;
            
            // Project down
            result += swiglu_out * w_down[inter * hidden_size + h_out];
        }
        
        output[idx] = result;
    }
}
// Host function

torch::Tensor swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor b_gate,
    torch::Tensor b_up
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
    
    auto output = torch::empty({batch_size, seq_len, intermediate_size}, options);
    auto gate_cache = torch::empty({batch_size, seq_len, intermediate_size}, options);
    auto up_cache = torch::empty({batch_size, seq_len, intermediate_size}, options);
    
    int total_elements = batch_size * seq_len * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    const void* b_gate_ptr = b_gate.defined() ? b_gate.data_ptr() : nullptr;
    const void* b_up_ptr = b_up.defined() ? b_up.data_ptr() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_forward_cuda",
        ([&] {
            swiglu_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                static_cast<const scalar_t*>(b_gate_ptr),
                static_cast<const scalar_t*>(b_up_ptr),
                output.data_ptr<scalar_t>(),
                gate_cache.data_ptr<scalar_t>(),
                up_cache.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor gate_cache,
    torch::Tensor up_cache
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto grad_x = torch::zeros_like(x);
    auto grad_w_gate = torch::zeros_like(w_gate);
    auto grad_w_up = torch::zeros_like(w_up);
    
    int total_elements = batch_size * seq_len * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_backward_cuda",
        ([&] {
            swiglu_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                gate_cache.data_ptr<scalar_t>(),
                up_cache.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                grad_w_gate.data_ptr<scalar_t>(),
                grad_w_up.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    return std::make_tuple(grad_x, grad_w_gate, grad_w_up);
}

torch::Tensor swiglu_down_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor b_gate,
    torch::Tensor b_up,
    torch::Tensor b_down
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = w_gate.size(1);
    
    auto output = torch::empty({batch_size, seq_len, hidden_size}, 
                               torch::TensorOptions()
                                   .dtype(x.dtype())
                                   .device(x.device()));
    
    int total_elements = batch_size * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    const void* b_gate_ptr = b_gate.defined() ? b_gate.data_ptr() : nullptr;
    const void* b_up_ptr = b_up.defined() ? b_up.data_ptr() : nullptr;
    const void* b_down_ptr = b_down.defined() ? b_down.data_ptr() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "swiglu_down_forward_cuda",
        ([&] {
            swiglu_down_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                w_gate.data_ptr<scalar_t>(),
                w_up.data_ptr<scalar_t>(),
                w_down.data_ptr<scalar_t>(),
                static_cast<const scalar_t*>(b_gate_ptr),
                static_cast<const scalar_t*>(b_up_ptr),
                static_cast<const scalar_t*>(b_down_ptr),
                output.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size, intermediate_size
            );
        })
    );
    
    return output;
}
