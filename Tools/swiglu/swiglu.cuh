#ifndef SWIGLU_FUSED_CUH
#define SWIGLU_FUSED_CUH

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA kernels
torch::Tensor swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor b_gate,
    torch::Tensor b_up
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor gate_output,
    torch::Tensor up_output
);

// Fused SwiGLU + Down projection
torch::Tensor swiglu_down_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor b_gate,
    torch::Tensor b_up,
    torch::Tensor b_down
);

#endif // SWIGLU_FUSED_CUH
