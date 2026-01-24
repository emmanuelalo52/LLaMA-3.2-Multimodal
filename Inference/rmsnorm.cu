#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "rmsnorm.cuh"

// Forward pass
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    // Robustly handle 2D or 3D tensors by flattening all but the last dimension
    const int N = input.numel() / input.size(-1); 
    const int C = input.size(-1);
    
    auto output = torch::empty_like(input);
    // Allocate RMS as float32 for high-precision backward pass
    auto rms = torch::empty({N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    // Allocate enough shared memory for warp reduction results
    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = sizeof(float) * ((BLOCK_SIZE + 31) / 32);

    kernels::rmsnorm_kernel_vectorized<at::Half, BLOCK_SIZE, true><<<N, BLOCK_SIZE, shared_mem>>>(
        reinterpret_cast<at::Half*>(output.data_ptr<at::Half>()),
        rms.data_ptr<float>(), 
        reinterpret_cast<const at::Half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        N, C, eps
    );

    return {output, rms};
}

// Backward pass
std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_out, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor rms) {
    
    const int N = input.numel() / input.size(-1);
    const int C = input.size(-1);
    
    auto d_input = torch::empty_like(input);
    auto d_weight = torch::zeros({C}, torch::dtype(torch::kFloat32).device(input.device()));

    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = sizeof(float) * ((BLOCK_SIZE + 31) / 32);

    kernels::rmsnorm_backward_kernel<at::Half, BLOCK_SIZE><<<N, BLOCK_SIZE, shared_mem>>>(
        reinterpret_cast<at::Half*>(d_input.data_ptr<at::Half>()),
        d_weight.data_ptr<float>(),
        reinterpret_cast<const at::Half*>(grad_out.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        rms.data_ptr<float>(),
        N, C
    );

    // Convert d_weight to match input dtype
    return {d_input, d_weight.to(input.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "RMSNorm Forward (CUDA)");
    m.def("backward", &rmsnorm_backward, "RMSNorm Backward (CUDA)");
}