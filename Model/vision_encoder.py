
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class VisionEncoderConfig:
    """Configuration for the plain ViT vision encoder."""

    def __init__(
        self,
        hidden_size: int = 1280,          # ViT-H default; 768 for ViT-B
        intermediate_size: int = 5120,    # FFN inner dim (4× hidden typical)
        num_hidden_layers: int = 32,      # ViT-H: 32, ViT-B: 12
        num_attention_heads: int = 16,    # ViT-H: 16, ViT-B: 12
        num_channels: int = 3,
        image_size: int = 560,            # Llama-3.2-Vision uses 560
        patch_size: int = 14,             # Llama-3.2-Vision uses 14
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,
        # projection_dim is injected by MLLAMAConfig (kept for API compat)
        projection_dim: Optional[int] = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        self.projection_dim = projection_dim


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------

class ViTPatchEmbeddings(nn.Module):
    """Convert image pixels → flat patch token sequence + positional encoding."""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Conv2d projects each patch to embed_dim in one step
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Learnable absolute positional embeddings (no CLS token — we want all patches)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).unsqueeze(0),  # [1, num_patches]
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        # patch_embedding: [B, embed_dim, H/P, W/P]
        x = self.patch_embedding(pixel_values)
        # flatten spatial dims: [B, embed_dim, num_patches]
        x = x.flatten(2)
        # transpose: [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        x = x + self.position_embedding(self.position_ids)
        return x


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (plain, no fancy tricks — SigLIP-free)
# ---------------------------------------------------------------------------

class ViTSelfAttention(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # reshape → [B, heads, N, head_dim]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        ctx = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(ctx)


# ---------------------------------------------------------------------------
# FFN (standard GELU — not SigLIP's variant)
# ---------------------------------------------------------------------------

class ViTMLP(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)   # standard GELU (not approximate="tanh" as in SigLIP)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class ViTEncoderBlock(nn.Module):
    """Pre-norm transformer block (standard ViT convention)."""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = ViTSelfAttention(config)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ViTMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention + residual
        hidden_states = hidden_states + self.self_attn(self.layernorm1(hidden_states))
        # Pre-norm FFN + residual
        hidden_states = hidden_states + self.mlp(self.layernorm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Full ViT Encoder Stack
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [ViTEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level Vision Encoder (drop-in for SiglipModel)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    Pure ViT feature extractor — no contrastive / sigmoid pair-wise loss.

    Input:  pixel_values  [B, C, H, W]
    Output: patch_embeds  [B, num_patches, hidden_size]

    This is a direct drop-in for `SiglipModel` used in `MllamaForConditionalGeneration`.
    The MultiModalProjector downstream handles mapping to the LLM hidden dim.
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        self.embeddings = ViTPatchEmbeddings(config)
        self.encoder = ViTEncoder(config)
        # Final layer-norm over patch dim (same as SigLIP tower)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: FloatTensor [B, C, H, W]

        Returns:
            FloatTensor [B, num_patches, hidden_size]
        """
        hidden_states = self.embeddings(pixel_values)      # [B, N, D]
        hidden_states = self.encoder(hidden_states)         # [B, N, D]
        hidden_states = self.post_layernorm(hidden_states)  # [B, N, D]
        return hidden_states
