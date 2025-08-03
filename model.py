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
LLAMA32_CONFIG = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

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

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    device = next(model.parameters()).device
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf'), device=logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


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
def compute_rope_params(head_dim,theta=10_000,context_length=4096,freq_config=None,dtype=torch.float32):
    assert head_dim % 2 == 0, "embedding dimension must be even"
    #inverse frequency
    iver_freq = 1.0/(theta **(torch.arange(0,head_dim,2,dtype=dtype)[: (head_dim//2)].float()/head_dim))
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
    postion = torch.arange(context_length,dtype=dtype)
    #angles
    angles = postion[:,None] * inv_freq[None,:]
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

#Feed forward
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"],cfg["hidden_dim"],dtype=["dtype"],bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"],cfg["hidden_dim"],dtype=["dtype"],bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"],cfg["emb_dim"],dtype=["dtype"],bias=False)
    def forward(self,x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        x = Functional.silu(fc1) * fc2
        return self.fc3(x)
    
    
#Group Query attention with KV cache
class GroupQueryAttention(nn.Module):
    def __init__(self,d_in,d_out,num_heads,num_kv_groups,kv_cache = False,dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "Dimension must be even"
        assert num_heads % num_kv_groups == 0, "they must be equal"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
    def forward(self,x,cos,sin,mask,past_kv=None,use_cache=False):
        b,num_tokens,d_in = x.shape
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

        #expand to maximum length
        keys = keys.repeat_interleave(self.group_size,dim=1)
        values = values.repeat_interleave(self.group_size,dim=1)

        if past_kv is not None:
            past_k, past_v = past_kv  # both: (b, h, past_s, d)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        #attention score
        attn_score = queries @ keys.transpose(2,3)
        # mask = self.mask[:num_tokens,:num_tokens].bool()
        attn_score = attn_score .masked_fill(mask,float("-inf"))
        
        attn_weight = torch.softmax(attn_score/(keys.shape[-1]**0.5),dim=-1)
        context_vector = (attn_weight @ values).transpose(1,2)
        context_vector = self.out_proj(context_vector)

        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = GroupQueryAttention(d_in=cfg["emb_dim"],d_out=cfg["emb_dim"],num_heads=cfg["n_heads"],num_kv_groups=cfg["n_kv_groups"],dtype=cfg["dtype"])
        self.norm1 = RMSNorm(cfg["emb_dim"]) 
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg)
    def forward(self, x, mask, cos, sin, past_kv=None, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        if use_cache:
            x, kv = self.attn(x, mask, cos, sin, past_kv=past_kv, use_cache=True)
        else:
            x = self.attn(x, mask, cos, sin)
            kv = None
        x = x + shortcut

        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx, past_kv=None, use_cache=False):
    # Embedding
        tok_embeds = self.tok_emb(in_idx)  # (b, seq_len, emb_dim)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        new_kv_cache = [] if use_cache else None
        for i, block in enumerate(self.trf_blocks):
            past_layer_kv = past_kv[i] if past_kv is not None else None
            x, updated_kv = block(x, mask, self.cos, self.sin, past_kv=past_layer_kv, use_cache=use_cache)
            if use_cache:
                new_kv_cache.append(updated_kv)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        if use_cache:
            return logits, new_kv_cache
        return logits
#Load Pretrained weights
def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    past_kv = None  # Initialize cache

    for step in range(max_new_tokens):
        if past_kv is None:
            idx_cond = idx[:, -context_size:]
        else:
            idx_cond = idx[:, -1:]

        with torch.no_grad():
            if past_kv is None:
                logits, past_kv = model(idx_cond, use_cache=True)
            else:
                logits, past_kv = model(idx_cond, past_kv=past_kv, use_cache=True)

        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(logits<min_val.unsqueeze(-1),torch.tensor(float("-inf"),device=logits.device))

        # Temperature / greedy decoding
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and (idx_next == eos_id).all():
            break
        # Append new token to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

#-----------------------------------------------------------------------------------Training---------------------------------------------------------------------------------------- 
if __name__ == "__main__":
    # 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
    from huggingface_hub import hf_hub_download, login

    # Optional: Use this for non-interactive login
    # login(token="your_huggingface_access_token")

    # Define model size string
    LLAMA_SIZE_STR = "1B"  # e.g. "8B", "70B", etc.

    # Login interactively (only if you're running the script manually)
    login()

    # Download tokenizer
    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )
    tokenizer = Tokenizer(tokenizer_file_path)
    if LLAMA_SIZE_STR == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename="model.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
        )
        combined_weights = load_file(weights_file)


    else:
        combined_weights = {}
        for i in range(1, 3):
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)


    model = Llama3Model(LLAMA32_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    model.to(device)