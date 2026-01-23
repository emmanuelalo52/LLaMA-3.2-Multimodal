#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "rmsnorm.cuh"

// Forward pass
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    const int N = input.size(0);
    const int C = input.size(1);
    auto output = torch::empty_like(input);
    
    // Calculate shared memory: (BLOCK_SIZE / 32) * sizeof(float)
    // We use 512 for BLOCK_SIZE to match our kernel templates
    size_t shared_mem = (512 / 32) * sizeof(float);

    kernels::rmsnorm_kernel_vectorized<at::Half, 512, false><<<N, 512, shared_mem>>>(
        reinterpret_cast<at::Half*>(output.data_ptr<at::Half>()),
        nullptr,
        reinterpret_cast<const at::Half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        N, C, eps
    );

    return output;
}

// Backward pass
std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_out, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor rms) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    
    auto d_input = torch::empty_like(input);
    // d_weight is float32 for atomicAdd accuracy
    auto d_weight = torch::zeros({C}, torch::dtype(torch::kFloat32).device(input.device()));

    size_t shared_mem = (512 / 32) * sizeof(float);

    kernels::rmsnorm_backward_kernel<at::Half, 512><<<N, 512, shared_mem>>>(
        reinterpret_cast<at::Half*>(d_input.data_ptr<at::Half>()),
        d_weight.data_ptr<float>(),
        reinterpret_cast<const at::Half*>(grad_out.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        rms.data_ptr<float>(),
        N, C
    );

    // Cast d_weight back to Half to match the model's expectations
    return {d_input, d_weight.to(torch::kHalf)};
}

// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "RMSNorm Forward (CUDA)");
    m.def("backward", &rmsnorm_backward, "RMSNorm Backward (CUDA)");
}