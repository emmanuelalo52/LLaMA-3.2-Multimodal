import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the compiled CUDA extension
try:
    import swiglu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: swiglu_fused CUDA extension not found. Falling back to PyTorch implementation.")
    print("To compile: python setup.py install")


class FusedSwiGLUFunction(torch.autograd.Function):
    """
    Autograd function for fused SwiGLU operation.
    
    Forward: output = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
    """
    
    @staticmethod
    def forward(ctx, x, w_gate, w_up, b_gate=None, b_up=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            w_gate: Gate weight matrix [hidden_size, intermediate_size]
            w_up: Up weight matrix [hidden_size, intermediate_size]
            b_gate: Gate bias [intermediate_size] (optional)
            b_up: Up bias [intermediate_size] (optional)
        
        Returns:
            output: [batch_size, seq_len, intermediate_size]
        """
        if CUDA_AVAILABLE and x.is_cuda:
            # Use fused CUDA kernel
            output = swiglu.forward(x, w_gate, w_up, b_gate, b_up)
            # Note: The CUDA kernel returns gate_cache and up_cache internally
            # We need to modify the kernel to return them for backward pass
            
            # For now, recompute for backward (we'll optimize this later)
            gate = F.linear(x, w_gate.t(), b_gate)
            up = F.linear(x, w_up.t(), b_up)
            
            ctx.save_for_backward(x, w_gate, w_up, gate, up)
        else:
            # Fallback to PyTorch implementation
            gate = F.linear(x, w_gate.t(), b_gate)
            up = F.linear(x, w_up.t(), b_up)
            output = F.silu(gate) * up
            ctx.save_for_backward(x, w_gate, w_up, gate, up)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SwiGLU.
        
        Args:
            grad_output: Gradient of loss w.r.t. output [batch_size, seq_len, intermediate_size]
        
        Returns:
            Gradients for (x, w_gate, w_up, b_gate, b_up)
        """
        x, w_gate, w_up, gate, up = ctx.saved_tensors
        
        if CUDA_AVAILABLE and x.is_cuda:
            # Use fused CUDA backward kernel
            grad_x, grad_w_gate, grad_w_up = swiglu.backward(
                grad_output, x, w_gate, w_up, gate, up
            )
            # Compute bias gradients
            grad_b_gate = grad_output.sum(dim=[0, 1]) if ctx.needs_input_grad[3] else None
            grad_b_up = grad_output.sum(dim=[0, 1]) if ctx.needs_input_grad[4] else None
        else:
            # Fallback PyTorch backward
            gate_activated = F.silu(gate)
            
            # Gradient through multiplication: d/d(gate_activated) and d/d(up)
            grad_gate_activated = grad_output * up
            grad_up = grad_output * gate_activated
            
            # Gradient through SiLU: d/d(gate)
            sigmoid_gate = torch.sigmoid(gate)
            grad_gate = grad_gate_activated * sigmoid_gate * (1 + gate * (1 - sigmoid_gate))
            
            # Gradients w.r.t. weights and inputs
            grad_x_gate = grad_gate @ w_gate
            grad_x_up = grad_up @ w_up
            grad_x = grad_x_gate + grad_x_up
            
            grad_w_gate = grad_gate.transpose(-2, -1) @ x
            grad_w_up = grad_up.transpose(-2, -1) @ x
            
            # Sum over batch and sequence for bias gradients
            grad_b_gate = grad_gate.sum(dim=[0, 1]) if ctx.needs_input_grad[3] else None
            grad_b_up = grad_up.sum(dim=[0, 1]) if ctx.needs_input_grad[4] else None
        
        return grad_x, grad_w_gate, grad_w_up, grad_b_gate, grad_b_up


class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU layer that combines gate and up projections.
    
    This is more efficient than separate Linear layers because it:
    1. Fuses the two matrix multiplications
    2. Applies SiLU activation inline
    3. Performs element-wise multiplication in the same kernel
    
    Usage:
        swiglu = FusedSwiGLU(hidden_size=4096, intermediate_size=11008)
        output = swiglu(x)  # x: [batch, seq_len, hidden_size]
    """
    
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate and up projections
        self.w_gate = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.w_up = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        
        if bias:
            self.b_gate = nn.Parameter(torch.zeros(intermediate_size))
            self.b_up = nn.Parameter(torch.zeros(intermediate_size))
        else:
            self.register_parameter('b_gate', None)
            self.register_parameter('b_up', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.w_gate, a=5**0.5)
        nn.init.kaiming_uniform_(self.w_up, a=5**0.5)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        
        Returns:
            output: [batch_size, seq_len, intermediate_size]
        """
        return FusedSwiGLUFunction.apply(x, self.w_gate, self.w_up, self.b_gate, self.b_up)
    
    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, bias={self.b_gate is not None}'


class FusedFeedForward(nn.Module):
    """
    Complete fused feedforward layer: SwiGLU + Down projection
    
    Architecture:
        x -> [Gate, Up] -> SiLU(Gate) * Up -> Down -> output
    
    This is equivalent to:
        gate = Linear(x)
        up = Linear(x)
        intermediate = SiLU(gate) * up
        output = Linear(intermediate)
    
    But implemented in fused CUDA kernels for efficiency.
    """
    
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Use fused SwiGLU
        self.swiglu = FusedSwiGLU(hidden_size, intermediate_size, bias=bias)
        
        # Down projection
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        intermediate = self.swiglu(x)
        output = self.w_down(intermediate)
        return output


def convert_feedforward_to_fused(feedforward_module):
    """
    Convert a standard FeedForward module to FusedFeedForward.
    
    This function helps migrate existing models to use fused kernels.
    
    Args:
        feedforward_module: Your existing FeedForward module with w1, w2, w3
    
    Returns:
        FusedFeedForward module with copied weights
    """
    hidden_size = feedforward_module.w2.in_features
    intermediate_size = feedforward_module.w1.out_features
    
    # Create fused module
    fused_ff = FusedFeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=feedforward_module.w1.bias is not None
    )
    
    # Copy weights
    # w1 = gate, w3 = up, w2 = down
    fused_ff.swiglu.w_gate.data.copy_(feedforward_module.w1.weight.data)
    fused_ff.swiglu.w_up.data.copy_(feedforward_module.w3.weight.data)
    fused_ff.w_down.weight.data.copy_(feedforward_module.w2.weight.data)
    
    if feedforward_module.w1.bias is not None:
        fused_ff.swiglu.b_gate.data.copy_(feedforward_module.w1.bias.data)
        fused_ff.swiglu.b_up.data.copy_(feedforward_module.w3.bias.data)
        fused_ff.w_down.bias.data.copy_(feedforward_module.w2.bias.data)
    
    return fused_ff