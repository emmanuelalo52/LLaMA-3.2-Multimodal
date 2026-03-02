import importlib.util
import torch
import torch.nn as nn

from vision_encoder import VisionEncoderConfig, VisionEncoder
from Tools.swiglu.FusedSwiglu import FusedSwiGLU

HAS_RMSNORM_EXT = importlib.util.find_spec("rmsnorm") is not None
if HAS_RMSNORM_EXT:
    rmsnorm = __import__("rmsnorm")

class KVCache:
    def __init__(self):
        self.key_cache: list = []
        self.value_cache: list = []

    def num_items(self) -> int:
        if not self.key_cache:
            return 0
        return self.key_cache[0].shape[-2]

    def update(self, key_states, value_states, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class LLAMA32Config:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 4096,
        context_length: int = 131072,
        n_heads: int = 32,
        n_layers: int = 16,
        hidden_dim: int = 8192,
        max_position_embeddings: int = 2048,
        n_kv_groups: int = 8,
        rope_base: float = 500000.0,
        rms_norm_eps: float = 1e-05,
        dtype=torch.float16,
        rope_freq=None,
        pad_token_index=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.context_length = context_length
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_kv_groups = n_kv_groups
        self.rope_base = rope_base
        self.rms_norm_eps = rms_norm_eps
        self.dtype = dtype
        self.rope_freq = rope_freq if rope_freq is not None else {
            "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_context_length": 8192,
        }
        self.pad_token_index = pad_token_index


class MLLAMAConfig:
    """Master config: VisionEncoderConfig (plain ViT) + LLAMA32Config."""

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index: int = -100,
        image_token_index: int = 128256,
        vocab_size: int = 128256,
        projection_dim: int = 4096,
        hidden_size: int = 4096,
        pad_token_index=None,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_index = pad_token_index

        if isinstance(vision_config, dict):
            self.vision_config = VisionEncoderConfig(**vision_config)
        else:
            self.vision_config = vision_config or VisionEncoderConfig()

        if isinstance(text_config, dict):
            self.text_config = LLAMA32Config(**text_config, pad_token_index=pad_token_index)
        else:
            self.text_config = text_config

        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class Linear_LORA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.linear.weight.requires_grad = False
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x):
        return self.linear(x) + (self.alpha / self.rank) * self.lora_b(self.lora_a(self.dropout(x)))


def repeat_kv(hidden_states, n_rep: int):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, residual=None):
        x = x.contiguous()
        weight = weight.contiguous()
        if residual is None:
            residual = torch.zeros_like(x)
        residual = residual.contiguous()
        output, rms = rmsnorm.forward(x, weight, residual, eps)
        ctx.save_for_backward(x, weight, rms)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rms = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_x, d_weight = rmsnorm.backward(grad_output, x, weight, rms)
        if d_weight.dtype != weight.dtype:
            d_weight = d_weight.to(weight.dtype)
        return d_x, d_weight, None, d_x


class LLAMARMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, residual=None):
        if HAS_RMSNORM_EXT and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return RMSNormFunction.apply(x, self.weight, self.eps, residual)
        if residual is not None:
            x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class LLAMARotaryEmbedding(nn.Module):
    def __init__(self, config: LLAMA32Config, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.n_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FusedFeedforward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.swiglu = FusedSwiGLU(hidden_size, intermediate_size, bias=bias)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.w_down(self.swiglu(x))


class GroupQueryAttention(nn.Module):
    def __init__(self, config: LLAMA32Config, layer_idx=None, dtype=None):
        super().__init__()
        assert config.hidden_size % config.n_heads == 0
        assert config.n_heads % config.n_kv_groups == 0
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups
        self.W_query = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.W_key   = nn.Linear(config.hidden_size, config.n_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(config.hidden_size, config.n_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype)
        self.rotary_emb = LLAMARotaryEmbedding(config)
        self.is_causal = True

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        b, num_tokens, _ = hidden_states.shape
        query = self.W_query(hidden_states).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key   = self.W_key(hidden_states).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        value = self.W_value(hidden_states).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if kv_cache is not None:
            key, value = kv_cache.update(key, value, self.layer_idx)
        key   = repeat_kv(key, self.group_size)
        value = repeat_kv(value, self.group_size)
        attn_score = query @ key.transpose(2, 3)
        if attention_mask is not None:
            attn_score = attn_score + attention_mask[:, :, :, : key.shape[-2]]
        attn_weight = torch.softmax(attn_score / (key.shape[-1] ** 0.5), dim=-1)
        ctx = (attn_weight @ value).transpose(1, 2).contiguous().reshape(b, num_tokens, -1)
        return self.out_proj(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, config: LLAMA32Config, layer_idx: int):
        super().__init__()
        self.att   = GroupQueryAttention(config, layer_idx=layer_idx, dtype=config.dtype)
        self.norm1 = LLAMARMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = LLAMARMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
        self.ff    = FusedFeedforward(config.hidden_size, config.hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        residual  = hidden_states
        normed    = self.norm1(hidden_states)
        attn_out  = self.att(normed, attention_mask=attention_mask,
                             position_ids=position_ids, kv_cache=kv_cache)
        # Fused add-norm (residual passed into RMSNorm kernel for CUDA fusion)
        normed_ff = self.norm2(attn_out, residual=residual)
        ff_out    = self.ff(normed_ff)
        return attn_out + ff_out

class MultiModalProjector(nn.Module):
    def __init__(self, config: MLLAMAConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )

    def forward(self, image_features):
        return self.linear(image_features)

class Llama3Model(nn.Module):
    def __init__(self, config: LLAMA32Config):
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_token_index
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=self.pad_token_id, dtype=config.dtype,
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )
        self.final_norm = LLAMARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.tok_emb

    def _prepare_attention_mask(self, attention_mask, hidden_state):
        bsz, seq_len, _ = hidden_state.shape
        device = hidden_state.device
        if attention_mask is None:
            base_mask = torch.ones((bsz, seq_len), device=device, dtype=hidden_state.dtype)
        elif attention_mask.dim() == 2:
            base_mask = attention_mask.to(device=device, dtype=hidden_state.dtype)
        elif attention_mask.dim() == 4:
            return attention_mask.to(device=device, dtype=hidden_state.dtype)
        else:
            raise ValueError("attention_mask must be 2D or 4D")
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_state.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        padding = ((1.0 - base_mask) * torch.finfo(hidden_state.dtype).min)[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
        return causal + padding

    def _prepare_position_ids(self, position_ids, hidden_state):
        if position_ids is not None:
            return position_ids.to(hidden_state.device)
        seq_len = hidden_state.shape[1]
        return torch.arange(seq_len, device=hidden_state.device).unsqueeze(0).expand(hidden_state.shape[0], -1)

    def forward(self, input_ids=None, input_embeds=None, attention_mask=None,
                position_ids=None, kv_cache=None):
        if input_embeds is not None:
            hidden_state = input_embeds
        elif input_ids is not None:
            hidden_state = self.tok_emb(input_ids)
        else:
            raise ValueError("Either input_ids or input_embeds must be provided")

        normalizer = torch.tensor(self.config.hidden_size ** 0.5,
                                  dtype=hidden_state.dtype, device=hidden_state.device)
        hidden_state = hidden_state * normalizer
        attention_mask = self._prepare_attention_mask(attention_mask, hidden_state)
        position_ids   = self._prepare_position_ids(position_ids, hidden_state)

        for block in self.trf_blocks:
            hidden_state = block(hidden_state, attention_mask=attention_mask,
                                 position_ids=position_ids, kv_cache=kv_cache)
        return self.final_norm(hidden_state)


class Llama3ForCausalLM(nn.Module):
    def __init__(self, config: LLAMA32Config):
        super().__init__()
        self.model = Llama3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, input_embeds=None, attention_mask=None,
                position_ids=None, kv_cache=None):
        outputs = self.model(input_ids=input_ids, input_embeds=input_embeds,
                             attention_mask=attention_mask, position_ids=position_ids,
                             kv_cache=kv_cache)
        return self.lm_head(outputs), kv_cache

    def tie_weights(self):
        self.lm_head.weight = self.model.tok_emb.weight


class MllamaForConditionalGeneration(nn.Module):
    """
    LLaMA-3.2 VLM — plain ViT vision encoder + LLaMA language model.

    Vision tower : VisionEncoder  (no SigLIP)
    Language model: Llama3ForCausalLM (GQA + RoPE + fused SwiGLU/RMSNorm .cu kernels)
    Bridge        : MultiModalProjector
    """

    def __init__(self, config: MLLAMAConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        self.ignore_index = config.ignore_index
        self.image_token_index = config.image_token_index

        # Vision: plain ViT, NOT SigLIP
        self.vision_model = VisionEncoder(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)

        # Language model
        self.language_model = Llama3ForCausalLM(config.text_config)

    def tie_weights(self):
        self.language_model.tie_weights()

    def get_input_embeddings(self):
        return self.language_model.model.tok_emb

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        image_mask=None,
        labels=None,
        kv_cache=None,
        **kwargs,
    ):
        image_features = None
        if pixel_values is not None:
            image_features = self.vision_model(pixel_values)           # [B, N, V_dim]
            image_features = self.multi_modal_projector(image_features) # [B, N, LM_dim]

        inputs_embeds = None
        if input_ids is not None:
            inputs_embeds = self.language_model.model.get_input_embeddings()(input_ids)

        if image_features is not None and inputs_embeds is not None:
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask
            )

        hidden_states = self.language_model.model(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        logits = self.language_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss, "hidden_states": hidden_states, "kv_cache": kv_cache}

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        batch_size, seq_len = input_ids.shape
        _, num_image_patches, _ = image_features.shape

        final_embeddings     = inputs_embeds.clone()
        final_attention_mask = (
            attention_mask.clone() if attention_mask is not None
            else torch.ones_like(input_ids)
        )
        image_token_mask = input_ids == self.image_token_index

        for batch_idx in range(batch_size):
            img_positions = torch.where(image_token_mask[batch_idx])[0]
            if len(img_positions) > 0:
                start_pos      = img_positions[0].item()
                end_pos        = min(start_pos + num_image_patches, seq_len)
                actual_patches = end_pos - start_pos
                final_embeddings[batch_idx, start_pos:end_pos] = image_features[batch_idx, :actual_patches]
                final_attention_mask[batch_idx, start_pos:end_pos] = 1

        return final_embeddings, final_attention_mask