import torch.nn as nn
import torch
import torch.nn.functional as Functional
import urllib
import zipfile
import os
from pathlib import Path
from safetensors.torch import load_file
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from siglip import SiglipVisionConfig,SiglipModel

class LLAMA32Config():
    def __init__(self,
                 vocab_size,
                 hidden_size=4096,
                 context_length=131072,
                 n_heads=32,
                 n_layers=16,
                 hidden_dim=8192, # Also called intermediate size
                 n_kv_groups=8,
                 rope_base=500000.0,
                 rms_norm_eps=1e-05,
                 dtype=torch.bfloat16,
                 rope_freq=None,
                 pad_token_index=None,
                 **kwargs,):
        super().__init__()
        self.vocab_size = vocab_size # Vocabulary size
        self.hidden_size = hidden_size # Hidden size
        self.context_length = context_length # Context length that was used to train the model
        self.n_heads = n_heads # Number of attention heads
        self.n_layers = n_layers # Number of layers
        self.hidden_dim = hidden_dim # Size of the intermediate dimension in FeedForward
        self.n_kv_groups = n_kv_groups # Key-Value groups for grouped-query attention
        self.rope_base = rope_base # The base in RoPE's "theta"
        self.rms_norm_eps = rms_norm_eps
        self.dtype = dtype # Lower-precision dtype to reduce memory usage. For lower GPUs use 'torch.float16'
        self.rope_freq = rope_freq if rope_freq is not None else { # RoPE frequency scaling
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
        self.pad_token_index=pad_token_index

class MLLAMAConfig():
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 ignore_index=-100,
                 image_token_index=128256,
                 vocab_size=128256,
                 projection_dim=2048,
                 hidden_size=2048,
                 pad_token_index=None,
                 **kwargs,
                 ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_index = pad_token_index

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config  

        self.text_config = LLAMA32Config(**text_config, pad_token_index=pad_token_index)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim




#Add LoRA
class Linear_LORA(nn.Module):
    def __init__(self,in_dim,out_dim,rank,alpha,dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim,bias=False)

        self.lora_a = nn.Linear(in_dim,rank,bias=False)
        self.lora_b = nn.Linear(in_dim,out_dim,bias=False)

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(p=dropout)

        #freeze original weights
        self.linear.weight.requires_grad=False
        self.lora_a.weight.requires_grad=True
        self.lora_b.weight.requires_grad=True
    def forward(self,x):
        frozen_out = self.linear(x)

        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        return frozen_out + (self.alpha /self.rank) * lora_out
    

def repeat_kv(hidden_states, n_rep):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    
#RMSNorm
class LLAMARMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class LLAMARotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(-2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Feed forward
class FeedForward(nn.Module):
    def __init__(self,config:LLAMA32Config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim,config.emb_dim,dtype=config.dtype,bias=False)
    def forward(self,x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        x = Functional.silu(fc1 * fc2) 
        return self.fc3(x)
    
    
#Group Query attention with KV cache
class GroupQueryAttention(nn.Module):
    def __init__(
            self,config:LLAMA32Config,layer_idx=None,dtype=None
    ):
        super().__init__()
        assert config.hidden_size % config.n_heads == 0, "d_out must be divisible by num_heads"
        assert config.n_heads % config.n_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        
        self.is_causal = True

        self.W_key = nn.Linear(config.hidden_size, config.n_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(config.hidden_size, config.n_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups

        self.W_query = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype)
        self.rotary_emb = LLAMARotaryEmbedding(dim=self.head_dim,max_position_embeddings=self.max_position_embeddings,base=self.rope_theta,)

    def forward(self,hidden_states,attention_mask=None,position_ids=None,kv_cache=None):
        b,num_tokens,_ = hidden_states.shape
        query = self.W_query(hidden_states)
        key = self.W_key(hidden_states)
        value = self.W_value(hidden_states)

        query = query.view(b,num_tokens,self.num_heads,self.head_dim)
        key = key.view(b,num_tokens,self.num_kv_groups,self.head_dim)
        value = value.view(b,num_tokens,self.num_kv_groups,self.head_dim)

        #Transpose
        queries = query.transpose(1,2)
        values = value.transpose(1,2)
        keys = key.transpose(1,2)

        #apply rope
        cos,sin = self.rotary_emb(values,position_ids,seq_len=None)
        queries,keys = apply_rotary_pos_emb(queries,keys,cos,sin)

        if kv_cache is not None:
            keys,queries = kv_cache.update(keys,queries,self.layer_idx)

        #expand to maximum length
        keys = repeat_kv(keys,self.group_size)
        values = repeat_kv(values,self.group_size)

        #attention score
        attn_score = queries @ keys.transpose(2,3)
        # mask = self.mask[:num_tokens,:num_tokens].bool()
        if attention_mask is not None:

            # Crop Attetion Mask to only include upto the number of key/value tokens we have to attend to 
            causal_mask = attention_mask[:, :, :, :keys.shape[-2]]

            # Add Causal Mask to score
            attn_score = attn_score + causal_mask
        
        attn_weight = torch.softmax(attn_score/(keys.shape[-1]**0.5),dim=-1)
        context_vector = (attn_weight @ values).transpose(1,2)
        context_vector = self.out_proj(context_vector)

        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self,config:LLAMA32Config,layer_idx):
        super().__init__()
        self.config= config
        self.att = GroupQueryAttention(config,layer_idx=layer_idx,dtype=config.dtype)
        self.norm1 = LLAMARMSNorm(dim=config.hidden_size,eps=config.rms_norm_eps) 
        self.norm2 = LLAMARMSNorm(dim=config.hidden_size,eps=config.rms_norm_eps)
        self.ff = FeedForward(config)
    def forward(self,hidden_states,attention_mask=None,position_ids=None,kv_cache=None,):
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states,_, = self.att(hidden_states=hidden_states,attention_mask=attention_mask,position_ids=position_ids,kv_cache=kv_cache,)

        hidden_states = residual+ hidden_states
        residual = hidden_states

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class MultiModalProjector(nn.Module):
    def __init__(self,config:MLLAMAConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size,config.vision_config.projection_dim)
    def forward(self,image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class Llama3Model(nn.Module):
    def __init__(self, config:LLAMA32Config):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = LLAMARMSNorm(config.hidden_size, eps=1e-5)
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)

        # Reusuable utilities
        cos, sin = compute_rope_params(
            head_dim=config.emb_dim // config.n_heads,
            theta_base=config.rope_base,
            context_length=config.context_length,
            freq_config=config.rope_freq
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = config

    def get_input_embeeddings(self):
        return self.tok_emb

    def forward(self, in_idx,kv_cache,input_embeds):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class MllamaForConditionalGeneration(nn.Module):
    def __init__(self)
