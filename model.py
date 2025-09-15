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
                 emb_dim,
                 context_length=131072,
                 n_heads=32,
                 n_layers=16,
                 hidden_dim=8192,
                 n_kv_groups=8,
                 rope_base=500000.0,
                 dtype=torch.bfloat16,
                 rope_freq=None,
                 pad_token_index=None,
                 **kwargs,):
        super().__init__()
        self.vocab_size = vocab_size # Vocabulary size
        self.emb_dim = emb_dim # Embedding dimension
        self.context_length = context_length # Context length that was used to train the model
        self.n_heads = n_heads # Number of attention heads
        self.n_layers = n_layers # Number of layers
        self.hidden_dim = hidden_dim # Size of the intermediate dimension in FeedForward
        self.n_kv_groups = n_kv_groups # Key-Value groups for grouped-query attention
        self.rope_base = rope_base # The base in RoPE's "theta"
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
                 pad_token_index=None
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


class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)
def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# def assign(left, right):
#     if left.shape != right.shape:
#         raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
#     return torch.nn.Parameter(torch.tensor(right)



    
#RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

#Rotary Postional Embedding 
def compute_rope_params(head_dim,theta_base=10_000,context_length=4096,freq_config=None,dtype=torch.float32,position_ids=None):
    assert head_dim % 2 == 0, "embedding dimension must be even"
    #inverse frequency
    inv_freq = 1.0/(theta_base **(torch.arange(0,head_dim,2,dtype=dtype)[: (head_dim//2)].float()/head_dim))
    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    #positons
    if position_ids is None:
        postion_ids = torch.arange(context_length,dtype=dtype)
    #angles
    angles = postion_ids[:,None] * inv_freq[None,:]
    #expand angles to match head_dim
    angles = torch.cat([angles,angles],dim=1)
    #sin and cos
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return cos, sin

#compute 2d RoPE
def rope(x,cos,sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    #d must be even
    head_dim % 2 == 0, "dimension must be even"
    x1 = x[...,:head_dim//2]
    x2 = x[...,head_dim//2:]
    cos = cos[:seq_len,:].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len,:].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2,x1),dim=-1)
    x_rotated = (x*cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

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
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
    def forward(self,x,cos,sin,mask,kv_cache=None):
        b,num_tokens,_ = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        query = query.view(b,num_tokens,self.num_heads,self.head_dim)
        key = key.view(b,num_tokens,self.num_kv_groups,self.head_dim)
        value = value.view(b,num_tokens,self.num_kv_groups,self.head_dim)

        #Transpose
        queries = query.transpose(1,2)
        values = value.transpose(1,2)
        keys = key.transpose(1,2)

        #apply rope
        keys = rope(keys,cos,sin)
        queries= rope(queries,cos,sin)

        if kv_cache is not None:
            keys,queries = kv_cache.update(keys,queries,self.layer_idx)

        #expand to maximum length
        keys = repeat_kv(keys,self.group_size)
        values = repeat_kv(values,self.group_size)

        #attention score
        attn_score = queries @ keys.transpose(2,3)
        # mask = self.mask[:num_tokens,:num_tokens].bool()
        attn_score = attn_score.masked_fill(mask,float("-inf"))
        
        attn_weight = torch.softmax(attn_score/(keys.shape[-1]**0.5),dim=-1)
        context_vector = (attn_weight @ values).transpose(1,2)
        context_vector = self.out_proj(context_vector)

        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self,config:LLAMA32Config):
        super().__init__()
        self.att = GroupQueryAttention(d_in=config.emb_dim,d_out=config.emb_dim,num_heads=config.n_heads,num_kv_groups=config.n_kv_groups,dtype=config.dtype)
        self.norm1 = RMSNorm(config.emb_dim) 
        self.norm2 = RMSNorm(config.emb_dim)
        self.ff = FeedForward(config)
    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin, mask)
        x = x + shortcut

        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

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

        self.final_norm = RMSNorm(config.emb_dim, eps=1e-5)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

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
# #Load Pretrained weights
# def assign(left, right, tensor_name="unknown"):
#     if left.shape != right.shape:
#         raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

#     if isinstance(right, torch.Tensor):
#         return torch.nn.Parameter(right.clone().detach())
#     else:
#         return torch.nn.Parameter(torch.tensor(right))


# def load_weights_into_llama(model, param_config, params):
#     model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

#     for l in range(param_config["n_layers"]):

#         # Load attention weights
#         model.trf_blocks[l].att.W_query.weight = assign(
#             model.trf_blocks[l].att.W_query.weight,
#             params[f"model.layers.{l}.self_attn.q_proj.weight"],
#             f"model.layers.{l}.self_attn.q_proj.weight"
#         )
#         model.trf_blocks[l].att.W_key.weight = assign(
#             model.trf_blocks[l].att.W_key.weight,
#             params[f"model.layers.{l}.self_attn.k_proj.weight"],
#             f"model.layers.{l}.self_attn.k_proj.weight"
#         )
#         model.trf_blocks[l].att.W_value.weight = assign(
#             model.trf_blocks[l].att.W_value.weight,
#             params[f"model.layers.{l}.self_attn.v_proj.weight"],
#             f"model.layers.{l}.self_attn.v_proj.weight"
#         )
#         model.trf_blocks[l].att.out_proj.weight = assign(
#             model.trf_blocks[l].att.out_proj.weight,
#             params[f"model.layers.{l}.self_attn.o_proj.weight"],
#             f"model.layers.{l}.self_attn.o_proj.weight"
#         )
#         model.trf_blocks[l].norm1.weight = assign(
#             model.trf_blocks[l].norm1.weight,
#             params[f"model.layers.{l}.input_layernorm.weight"],
#             f"model.layers.{l}.input_layernorm.weight"
#         )

#         # Load FeedForward weights
#         model.trf_blocks[l].ff.fc1.weight = assign(
#             model.trf_blocks[l].ff.fc1.weight,
#             params[f"model.layers.{l}.mlp.gate_proj.weight"],
#             f"model.layers.{l}.mlp.gate_proj.weight"
#         )
#         model.trf_blocks[l].ff.fc2.weight = assign(
#             model.trf_blocks[l].ff.fc2.weight,
#             params[f"model.layers.{l}.mlp.up_proj.weight"],
#             f"model.layers.{l}.mlp.up_proj.weight"
#         )
#         model.trf_blocks[l].ff.fc3.weight = assign(
#             model.trf_blocks[l].ff.fc3.weight,
#             params[f"model.layers.{l}.mlp.down_proj.weight"],
#             f"model.layers.{l}.mlp.down_proj.weight"
#         )
#         model.trf_blocks[l].norm2.weight = assign(
#             model.trf_blocks[l].norm2.weight,
#             params[f"model.layers.{l}.post_attention_layernorm.weight"],
#             f"model.layers.{l}.post_attention_layernorm.weight"
#         )

    # # Load output layer weights
    # model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    # if "lm_head.weight" in params.keys():
    #     model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    # else:
    #     model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    #     print("Model uses weight tying.")


#-----------------------------------------------------------------------------------Training---------------------------------------------------------------------------------------- 
# if __name__ == "__main__":
#     # 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
#     from huggingface_hub import hf_hub_download, login

#     # Optional: Use this for non-interactive login
#     # login(token="your_huggingface_access_token")

#     # Define model size string
#     LLAMA_SIZE_STR = "1B"  # e.g. "8B", "70B", etc.

#     # Login interactively (only if you're running the script manually)
#     login()

#     # Download tokenizer
#     tokenizer_file_path = hf_hub_download(
#         repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}",
#         filename="original/tokenizer.model",
#         local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}"
#     )
#     tokenizer = Tokenizer(tokenizer_file_path)
#     if LLAMA_SIZE_STR == "1B":
#         weights_file = hf_hub_download(
#             repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}",
#             filename="model.safetensors",
#             local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}"
#         )
#         combined_weights = load_file(weights_file)


#     else:
#         combined_weights = {}
#         for i in range(1, 3):
#             weights_file = hf_hub_download(
#                 repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}",
#                 filename=f"model-0000{i}-of-00002.safetensors",
#                 local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}"
#             )
#             current_weights = load_file(weights_file)
#             combined_weights.update(current_weights)


#     model = Llama3Model(LLAMA32_CONFIG)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


#     model.to(device)