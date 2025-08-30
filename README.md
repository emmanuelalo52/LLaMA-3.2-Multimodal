# 🦙 LLaMA‑3.2 Multimodal (WIP) — Deep Technical README
**(Text backbone + LoRA + RoPE + GQA) + SigLIP Vision (scaffolded)**

> **Status:** 🚧 **Active development.**  
> The **text backbone** (tokenizer, model blocks, RoPE, GQA attention, KV‑cache, weight loading, and an autoregressive generator) is implemented and runnable for inference.  
> The **vision stack (SigLIP)** and **image preprocessing** are implemented as separate modules; multimodal fusion (vision → text) is documented and example code is provided, but **not fully wired** into the LLaMA runtime yet.

---

# Table of contents
1. [Quick Start (TL;DR)](#quick-start-tldr)  
2. [Repository Contents & File map](#repository-contents--file-map)  
3. [High‑level Architecture](#high-level-architecture)  
4. [Text Backbone — Detailed Component Reference](#text-backbone---detailed-component-reference)  
   - Configuration (`LLAMA32_CONFIG`)  
   - Tokenizer (tiktoken wrapper)  
   - Embedding layer & shapes  
   - RoPE (math + implementation notes)  
   - Grouped‑Query Attention (GQA) & KV‑cache (shapes + behavior)  
   - FeedForward (SwiGLU)  
   - RMSNorm  
   - TransformerBlock & Llama3Model forward API  
   - LoRA (`Linear_LORA`) — design and wiring guidance  
   - Weight loading (`load_weights_into_llama`) mapping  
   - Generation (two variants in `model.py`)  
5. [Vision Stack (SigLIP) — Detailed Reference](#vision-stack-siglip---detailed-reference)  
   - Config and patch embedding math  
   - Attention implementation notes and bug fixes  
   - Encoder/Transformer forward notes  
6. [Image Preprocessing (`processing_siglip.py`)](#image-preprocessing-processing_siglippy)  
   - Transform pipeline, defaults, and shapes  
   - `paligemmaProcessor` usage and corrections  
7. [Multimodal Integration — Design Patterns & Example Code](#multimodal-integration---design-patterns--example-code)  
   - Option A: Prepend visual tokens (recommended starting point)  
   - Option B: Cross‑attention fusion (higher fidelity)  
   - Projection & tokenization recipe (code sample)  
8. [Training / Fine‑tuning Recipes (LoRA & adapters)](#training--fine-tuning-recipes-lora--adapters)  
   - Example LoRA-only training loop  
   - Saving / Loading adapter weights  
9. [Debugging & Known Issues (Concrete)](#debugging--known-issues-concrete)  
10. [Testing Checklist & Unit Tests to add](#testing-checklist--unit-tests-to-add)  
11. [Performance, Memory & Precision guidance](#performance-memory--precision-guidance)  
12. [Contributing, License & Citation](#contributing-license--citation)  
13. [Changelog (what's newly added from your uploaded files)](#changelog-whats-newly-added-from-your-uploaded-files)  

---

# Quick Start (TL;DR)

Install the runtime dependencies:
```bash
pip install --upgrade pip
pip install torch safetensors tiktoken huggingface_hub pillow numpy
```

Minimal inference flow (text-only):
```python
from model import Tokenizer, Llama3Model, LLAMA32_CONFIG, load_weights_into_llama, generate
from safetensors.torch import load_file
import torch

# 1) Tokenizer (path from HF hub)
tokenizer = Tokenizer("path/to/original/tokenizer.model")

# 2) Load safetensors (after HF login)
params = load_file("path/to/model.safetensors")

# 3) Build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Llama3Model(LLAMA32_CONFIG).to(device)

# 4) Assign weights
load_weights_into_llama(model, LLAMA32_CONFIG, params)
model.eval()

# 5) Encode and run generate (KV-cache aware generate)
prompt = "### Instruction:\nWrite a short motivating sentence about persistence.\n\n### Response:\n"
ids = tokenizer.encode(prompt, bos=True, eos=False)
idx = torch.tensor(ids, device=device).unsqueeze(0)
out = generate(model, idx, max_new_tokens=64, context_size=LLAMA32_CONFIG["context_length"],
               temperature=0.8, top_k=50)
print(tokenizer.decode(out.squeeze(0).tolist()))
```

Vision quick example (SigLIP):
```python
from siglip import SiglipVisionConfig, SiglipModel
import torch
config = SiglipVisionConfig()
vision = SiglipModel(config)
dummy = torch.randn(1, 3, 224, 224)
patch_embeds = vision(dummy)   # (1, num_patches, hidden_size)
```

---

# Repository Contents & File map
- `model.py` — **Text backbone**: tokenizer wrapper, LLAMA config, RoPE, RMSNorm, FeedForward, Grouped‑Query Attention (GQA), Transformer / LLaMA model, LoRA wrapper, safetensors loader, and two `generate` functions (simple & KV‑cache aware).
- `siglip.py` — **Vision stack** (SigLIP): `SiglipVisionConfig`, patch embedding (Conv2d projection), attention, MLP, encoder stack, and SiglipModel wrapper.
- `processing_siglip.py` — Image processing utilities and `paligemmaProcessor` (tokenizer + image preprocessor).
- `examples/` — (suggested) `text_inference.py`, `vision_inference.py`, `multimodal_demo.py` (you can add these as you iterate).
- `README.md` — this file.

---

# High‑level Architecture

**Goal:** Provide a compact, readable LLaMA‑3.2–style text backbone and a modular vision encoder that you can connect to the tokenizer/model to make a multimodal system.

Core ideas:
- **Text-only backbone** remains fully functional and can decode tokens autoregressively (with or without KV‑cache).
- **Vision encoder** (SigLIP) maps images → patch embeddings. Project these embeddings into the LLaMA token embedding space and **prepend or inject** them as "visual tokens".
- **LoRA adapters** allow efficient fine‑tuning by adding low‑rank updates to linear layers while keeping base weights frozen.

Below is an ASCII overview of inference flow:

```
[ image(s) ] -> preprocessing_siglip -> pixel_values -> SigLIP -> patch_embeds -> projection -> visual_token_embeds
                                                                     |
                                                                     v
[text prompt tokens] -> Tokenizer -> token_ids -> token_embeddings
                                                                     |
                                                                     v
[visual_token_embeds + token_embeddings] -> Llama3Model -> logits -> sample -> token_ids
```

---

# Text Backbone — Detailed Component Reference

This section documents internals, shapes, and implementation notes for the text backbone defined mostly in `model.py`.

## LLAMA32_CONFIG (fields)
The model config in `model.py`:
```py
LLAMA32_CONFIG = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Maximum context (designed for long contexts)
    "emb_dim": 2048,                 # Token embedding dimension
    "n_heads": 32,                   # Attention heads
    "n_layers": 16,                  # Transformer blocks
    "hidden_dim": 8192,              # FFN intermediate hidden size
    "n_kv_groups": 8,                # Number of KV groups for Grouped Query Attention
    "rope_base": 500_000.0,          # RoPE base (theta)
    "dtype": torch.bfloat16,         # Default dtype for weights (if available)
    "rope_freq": {                   # RoPE frequency scaling config
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}
```

### Why these values?
- `n_kv_groups` implements **GQA**, trading a smaller KV memory footprint for repeated queries across groups of heads. If `n_heads=32` and `n_kv_groups=8`: `group_size = n_heads // n_kv_groups = 4` (each KV vector serves 4 query heads).
- `rope_base` and `rope_freq` are tuned to support long contexts via frequency scaling (see RoPE section).

---

## Tokenizer (class `Tokenizer`)
- Thin wrapper around `tiktoken` BPE that loads a tokenizer model file from `original/tokenizer.model`.
- Adds Meta LLaMA special token IDs (hardcoded in `model.py`).
- API:
  - `Tokenizer(model_path)` — raise `FileNotFoundError` if `model_path` missing.
  - `.encode(text, bos=False, eos=False) -> List[int]` — optionally prepend BOS and append EOS special IDs.
  - `.decode(ids) -> str`

**Note:** The implementation constructs `tiktoken.Encoding(...)` using `load_tiktoken_bpe(...)` result. If you get tokenization mismatches with HF weights, ensure you downloaded the same `tokenizer.model` file from the HF repo matching the weights.

---

## Embedding & Token shapes
- Embedding matrix shape: `(vocab_size, emb_dim)` — e.g. `(128256, 2048)`
- Tokenized input `idx` should be a 2D tensor `(batch, seq_len)` of dtype `torch.long`.
- Token embedding lookup yields `tok_embeds`: `(batch, seq_len, emb_dim)`.

---

## RoPE — Rotary Positional Embeddings
### Implementation functions:
- `compute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None, dtype=torch.float32)`  
  Returns `(cos, sin)` arrays shaped `(context_length, head_dim)` (since angles are duplicated).
- `rope(x, cos, sin)`  
  Takes `x` with shape `(batch, heads, seq_len, head_dim)` and applies rotary rotation using the provided `cos` and `sin`.

### Mathematical summary
For a head dimension `d = head_dim` (must be even), RoPE interleaves pairs of dimensions and rotates them by angle `θ_i * pos` where θ_i = inv_freq[i] and `pos` is token position.

The simplified pairwise rotation on a pair `(x_{2i}, x_{2i+1})` is:
```
[x'] = [ cos * x - sin * y ]
[y']   [ sin * x + cos * y ]
```
Implementation note: `compute_rope_params` constructs a vector of angles with frequency scaling to allow extrapolation beyond training context.

### Known requirements & gotchas
- `head_dim` must be even.
- `compute_rope_params` uses a `freq_config` to alter `inv_freq`. Validate shapes for `angles`, `sin`, and `cos`.
- In `model.py`, `cos` and `sin` are registered as buffers (non-persistent) and later sliced up to `seq_len` in `rope`.

---

## Grouped Query Attention (GQA)
**Motivation:** share keys/values between groups of query heads to reduce KV memory while keeping many query heads.

### Key variables & shapes
- `d_out` (embedding dim) = `emb_dim`.
- `num_heads` (`H`) → number of query heads (e.g., 32).
- `num_kv_groups` (`G`) → number of distinct KV groups (e.g., 8).
- `head_dim` = `d_out // num_heads` (embedding per head).
- `group_size` = `num_heads // num_kv_groups`.

### Projections
- Queries: `W_query` produces `(batch, seq_len, d_out)` → reshaped to `(batch, seq_len, H, head_dim)` → transposed to `(batch, H, seq_len, head_dim)`.
- Keys: `W_key` outputs `(batch, seq_len, G, head_dim)` → transposed to `(batch, G, seq_len, head_dim)`.
- Values: `W_value` similarly `(batch, G, seq_len, head_dim)`.

### Expansion
Keys/values are repeated to match `H`:
```py
keys = keys.repeat_interleave(group_size, dim=1)   # (batch, H, seq_len, head_dim)
values = values.repeat_interleave(group_size, dim=1)
```
Then standard scaled dot-product attention using per‑head queries.

### KV‑cache (inference speed)
- Past KV for each layer is expected as `(past_k, past_v)` with shapes `(batch, H, past_seq, head_dim)` — the code presently uses `(b, h, past_s, d)` notation.
- At generation, when `past_kv` is provided, the code concatenates along sequence dimension to append new keys/values.

**Implementation notes:**
- `masked_fill` uses a boolean `mask` — ensure `mask` dtype is `bool` and it is broadcastable to attention scores `(batch, H, seq_len, seq_len)`.
- The code currently constructs `mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)` — which is correct for causal masking.

**Potential bug to fix:** in current `model.py` the variable `k` and `v` are referenced but not defined (should be `keys`/`values` after transforms). See the Known Issues section for exact lines.

---

## FeedForward (SwiGLU‑style)
- Implemented as:
  - `fc1 = Linear(emb_dim, hidden_dim)` → `silu(fc1(x))`
  - `fc2 = Linear(emb_dim, hidden_dim)`  (this looks like a bug; typically SwiGLU uses two projections from `emb_dim` to `hidden_dim` but then multiplies them and a final projection back)
  - `fc3 = Linear(hidden_dim, emb_dim)`

**Recommended fix:** `fc2` should be `Linear(cfg["hidden_dim"], cfg["hidden_dim"])` is not standard — instead follow canonical SwiGLU:
```py
fc_gate = Linear(emb_dim, hidden_dim)
fc_up   = Linear(emb_dim, hidden_dim)
fc_down = Linear(hidden_dim, emb_dim)
out = fc_down(silu(fc_gate(x)) * fc_up(x))
```
But the shipped code uses `fc1`, `fc2`, `fc3`; review and correct dims if you fine‑tune.

---

## RMSNorm
- Implemented as `RMSNorm(dim, eps=1e-6)`.
- Internal `_norm(x)` computes `x * rsqrt(mean(x^2) + eps)` per last dimension and scales by learnable `weight`.
- Note: Implementation converts x to `.float()` for numerical stability and then casts back.

---

## TransformerBlock & Llama3Model
- `TransformerBlock(cfg)` wraps `GroupQueryAttention`, `RMSNorm` (two layers), and `FeedForward`.
- `TransformerBlock.forward(x, mask, cos, sin, past_kv=None, use_cache=False)` should:
  - Apply `norm1` → attention (may return `(context, kv)` when caching) → residual add.
  - Apply `norm2` → feedforward → second residual add.
  - **Important:** the file's `TransformerBlock.forward` *does not currently return `(x, kv)`* in a consistent way. If `use_cache` is `True`, it should return `(x, (k,v))` for the layer so `Llama3Model` can collect them.

- `Llama3Model.forward(in_idx, past_kv=None, use_cache=False)`:
  - Converts `in_idx` `(b, seq_len)` → embeddings `(b, seq_len, emb_dim)`.
  - Builds causal mask: `(seq_len, seq_len)` boolean upper-triangular.
  - Iterates blocks and collects per-layer `kv` caches when `use_cache`.
  - Applies final norm and outputs logits `(b, seq_len, vocab_size)`.

**Device & dtype:** final logits are cast to `cfg["dtype"]` for compatibility with LM head.

---

## LoRA (class `Linear_LORA`)
- `Linear_LORA(in_dim, out_dim, rank, alpha, dropout)` provides:
  - Frozen base linear: `self.linear` with `requires_grad=False`.
  - Trainable low-rank factors `lora_a: in_dim -> rank` and `lora_b: rank -> out_dim`.
  - `forward(x) = linear(x) + (alpha / rank) * lora_b(lora_a(dropout(x)))`

**How to wire LoRA into model:**
1. For each target `nn.Linear` you want to adapt (e.g., `att.W_query`, `att.out_proj`, `ffn` projections), replace it with an instance of `Linear_LORA` while copying the original weights into `.linear`.
2. Alternatively, create a wrapper that keeps references to both original and lora modules and toggles forward behavior based on `train_lora = True`.

**Saving adapters:** save only `lora_a` and `lora_b` weights to persist adapters.

---

## Weight loading: `load_weights_into_llama(model, param_config, params)`
- Purpose: map safetensors keys (HF converged weights) into the PyTorch model.
- Expected param key names (examples):
  - `model.embed_tokens.weight` → `model.tok_emb.weight`
  - `model.layers.{i}.self_attn.q_proj.weight` → `model.trf_blocks[i].att.W_query.weight`
  - `model.layers.{i}.mlp.gate_proj.weight` → `model.trf_blocks[i].ff.fc1.weight` (or as mapped)
  - `model.norm.weight` → `model.final_norm.weight`
  - `lm_head.weight` or fallback to `model.embed_tokens.weight`
- `assign(left, right, tensor_name)` validates shapes and wraps tensors as `torch.nn.Parameter`.

**Multi‑shard checkpoint handling:** The script demonstrates combining shards into a single `combined_weights` dict when the checkpoint is sharded.

**Common shape mismatch cause:** mismatched ordering (transpose) — some HF safetensors use shapes transposed relative to your `nn.Linear.weight` expectations. If you see shape mismatches, try `.T` on the weight or inspect the HF checkpoint naming conventions.

---

## Generation APIs
Two variants exist in `model.py`:

1. **Simple `generate`** (no KV-cache): iteratively runs the model on the full prefix and samples next tokens. Good for short sequences and debugging.

2. **KV‑cache aware `generate`** (recommended for autoregressive inference):  
   - On first step, run model with full context and receive `(logits, past_kv)`.  
   - On subsequent steps, pass only the newly generated token and `past_kv` to get updated KV arrays; the model concatenates past keys/values internally.
   - This saves computation and memory across long generated sequences.

**Top‑k implementation note:** the provided implementations use `torch.topk` and threshold tokens by replacing logits below the kth top with −inf. Implementation must use `values = topk_values[:, -1]` to get the kth highest value and compare to logits.

**Sampling options:** `temperature`, `top_k`, and greedy (temperature=0) are supported.

---

# Vision Stack (SigLIP) — Detailed Reference

Files: `siglip.py` and `processing_siglip.py`.

## `SiglipVisionConfig`
Key defaults (from `siglip.py`):
- `hidden_size=768`
- `intermediate_size=12` *(likely a bug: intermediate_size too small — should be a multiple of hidden_size, e.g., 3072)*
- `num_hidden_layers=12`
- `num_attention_heads=12`
- `num_channels=3`
- `image_size=224`
- `patch_size=16`
- `layer_norm_eps=1e-6`
- `num_image_tokens` optional for multimodal integration

**Action:** Review `intermediate_size` default and set to `hidden_size * 4` (common practice) unless intentionally small.

## Patch embedding
`SiglipVisionEmbedding` uses `nn.Conv2d` with `kernel_size=patch_size, stride=patch_size`:
- Input shape: `(B, C, H, W)`
- Output of conv: `(B, embed_dim, H/patch, W/patch)` where `num_patches = (image_size // patch_size) ** 2`.
- Flatten to `(B, num_patches, embed_dim)` and add position embeddings.

## Attention (SiglipAttention) — corrections
The shipped implementation has a few coding bugs. Key corrections:

1. Projection calls:
```py
keys_state  = self.k_proj(hidden_states)
query_state = self.q_proj(hidden_states)
value_state = self.v_proj(hidden_states)
```
(Instead of `self.k_proj.view(hidden_states)`)

2. After projection, reshape:
```py
keys_state = keys_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
```

3. Scale factor sign: usually `scores = (query @ key.transpose(-2, -1)) / sqrt(head_dim)`

4. Softmax dtype: keep softmax in `float32` for numerical stability, then cast back to original dtype.

5. `nn.functional.gelu(hidden_state, approximate="tanh")` — `approximate="tanh"` is a valid argument only in recent PyTorch versions; prefer `nn.functional.gelu(hidden_state)` for portability.

6. `SiglipVisionTransformer.forward` currently contains multiple mistakes (e.g., `hidden_states = self.embedding` rather than `self.embedding(pixel_values)`). See corrections below.

**Corrected simplified forward (vision transformer):**
```py
def forward(self, pixel_values):
    patch_embeds = self.embedding(pixel_values)         # (B, num_patches, embed_dim)
    encoded = self.encoder(patch_embeds)               # (B, num_patches, embed_dim)
    last_hidden_state = self.layernorm(encoded)
    return last_hidden_state
```

---

# Image Preprocessing (`processing_siglip.py`)

Utilities:
- `resize(image, size=(H,W), resample=Image.Resampling.BICUBIC, reducing_gap=None)` → returns PIL image resized.
- `rescale(image, scale=1/255.0)` → multiplies pixel values then casts dtype.
- `normalize(image, mean, std)` → elementwise `(image - mean)/std`.
- `process_images(images, size, rescale_factor, image_mean, image_std)`  
  Steps: resize → np.array → scale → normalize → transpose to `[C,H,W]` → stack into tensor `(B, C, H, W)`.

`paligemmaProcessor` class:
- Intended to add an `<image>` special token to tokenizer and map a number of image tokens.
- **Bugs / fixes:**
  - `IMAGE_TOKEN` defined locally but `self.IMAGE_TOKEN` used later — pick one. Fix: set `self.IMAGE_TOKEN = "<image>"`.
  - `tokens_to_add = {"additional_special tokens":[self.IMAGE_TOKEN]}` contains a space in the key and should be `"additional_special_tokens": [self.IMAGE_TOKEN]` for HuggingFace tokenizers. If using `tiktoken`, use the token addition API accordingly.
  - `self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)` — for tiktoken this API differs; you may need to use `tokenizer.special[...]` mapping or `tokenizer.model.token_to_id(...)` depending on tokenizer implementation.
  - BOS/EOS flags are toggled by setting `tokenizer.add_bos_token` — this depends on tokenizer object: for HF tokenizers, use `add_special_tokens` and `tokenizer.bos_token_id`.

**Output shape:** The processor returns `{"pixel_value": Tensor[B,C,H,W], "input_ids": Tensor[B, L], "attention_mask": Tensor[B, L]}` (key names may vary in code; ensure keys match model expectations).

---

# Multimodal Integration — Design Patterns & Example Code

Below are practical strategies and a runnable example to project vision outputs into LLM embeddings and prepend them as visual tokens.

## Strategy A — Prepend visual tokens (simplest)
1. Process raw images to `pixel_values = process_images([...])` → `(B, C, H, W)`.
2. Forward through SigLIP: `patch_embeds = vision_model(pixel_values)` → `(B, P, V_dim)` where `P = num_patches`.
3. Convert `patch_embeds` into `image_seq_len` tokens:
   - Option 1: average pool into `image_seq_len` groups: e.g., `torch.nn.AdaptiveAvgPool1d(image_seq_len)` over `P`.
   - Option 2: learnable projection `nn.Conv1d` or `nn.Linear` that maps `P` → `image_seq_len`.
4. Project visual token vectors from `V_dim` → `emb_dim` of the LLaMA model:
```py
proj_visual = nn.Linear(V_dim, emb_dim)
visual_token_embeds = proj_visual(pooled_patch_embeds)  # (B, image_seq_len, emb_dim)
```
5. Build textual embeddings with tokenizer:
```py
text_ids = tokenizer.encode(prompt, bos=True, eos=False)  # list
text_embeds = model.tok_emb(torch.tensor([text_ids], device=device))
```
6. Concatenate:
```py
combined_embeds = torch.cat([visual_token_embeds, text_embeds], dim=1)  # (B, image_seq_len + seq_len, emb_dim)
```
7. Supply combined embeddings to transformer stacks (requires small change: current `Llama3Model.forward` expects token ids and does `self.tok_emb(in_idx)`; change to accept `inputs_embeds` directly or provide `fake token ids` whose embedding matrix is replaced for the visual token positions).

**Code example (sketch):**
```py
# Given model: Llama3Model(LLAMA32_CONFIG)
vision = SiglipModel(vision_config)
pixel_values = processor(... )["pixel_values"].to(device)
with torch.no_grad():
    patch_embeds = vision(pixel_values)   # (B, P, Vdim)

# Pool patches to image_seq_len
pooler = torch.nn.AdaptiveAvgPool1d(image_seq_len)
pooled = pooler(patch_embeds.transpose(1,2)).transpose(1,2)  # (B, image_seq_len, Vdim)

# Project to LLaMA emb_dim
proj = torch.nn.Linear(Vdim, LLAMA32_CONFIG["emb_dim"]).to(device)
visual_token_embeds = proj(pooled)  # (B, image_seq_len, emb_dim)

# Text token ids -> embeddings
ids = tokenizer.encode(prompt, bos=True)
text_ids = torch.tensor([ids], device=device)
text_embeds = model.tok_emb(text_ids)  # (B, seq_len, emb_dim)

# Concatenate and run transformer: you must extend Llama3Model to accept inputs_embeds
combined = torch.cat([visual_token_embeds, text_embeds], dim=1)
logits = model.forward_with_embeds(combined)   # implement this helper
```

## Strategy B — Cross‑attention fusion (higher fidelity)
Insert cross‑attention modules into the LLaMA stack so that text queries can attend to visual keys/values. This requires:
- Keeping vision patch embeddings as K/V (projected into same head_dim and grouped if using GQA).
- Modifying TransformerBlock to optionally run a cross-attn pass where `q` comes from text, `k,v` from vision patches.

**Tradeoffs:** Better multimodal alignment; more complex and heavier compute.

---

# Training / Fine‑tuning Recipes

## LoRA-only adapter training (recommended for initial experiments)
- Freeze base model parameters.
- Replace selected `nn.Linear` modules (e.g., `W_query`, `out_proj`, `ffn` down/up projections) with `Linear_LORA` instances, or alternatively keep base modules and add LoRA modules in parallel and sum outputs at forward time.
- Optimizer: `AdamW` on LoRA parameters only.
- Typical hyperparameters (example):
  - `rank=8` or `rank=16` (smaller ranks lower memory but reduce capacity)
  - `alpha=32` (scaling)
  - `lr = 1e-4` (warmup 100–1000 steps), weight_decay=0.0
  - batch_size depends on GPU memory; use gradient accumulation
  - Use mixed precision (AMP) and gradient checkpointing when necessary

### Example training loop (pseudo-code)
```py
# Assume model has been modified to include lora modules with .parameters() trainable
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)    # ensure model returns loss or compute CrossEntropyLoss
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
```

## Saving & Loading LoRA adapters
- Save `lora_a` and `lora_b` weights for every replaced layer:
```py
adapter_state = {}
for name, module in model.named_modules():
    if isinstance(module, Linear_LORA):
        adapter_state[f"{name}.lora_a.weight"] = module.lora_a.weight.cpu()
        adapter_state[f"{name}.lora_b.weight"] = module.lora_b.weight.cpu()
torch.save(adapter_state, "lora_adapters.pth")
```

- To load, instantiate same adapter modules and `load_state_dict` using these keys.

---

# Debugging & Known Issues (Concrete)

Below are *file- and function-level* problems found in the uploaded code and suggested fixes — crucial when you try to run/train this code.

### `model.py` issues and fixes
1. **Duplicate `assign` function.** `assign` is defined twice. Keep the latter that validates shapes and returns `torch.nn.Parameter(...)`. Remove or rename the other.
2. **`FeedForward` dim mismatch.** `fc2` currently constructed as `nn.Linear(cfg["emb_dim"], cfg["hidden_dim"])` — canonical SwiGLU uses `fc_gate` and `fc_up` both mapping `emb_dim -> hidden_dim`. The code also names `fc3` mapping back from `hidden_dim -> emb_dim`. Rework naming for clarity.
3. **`TransformerBlock.forward` not returning values.** When `use_cache` is True, `attn` should return both `context_vector` and `(k,v)`; `TransformerBlock.forward` should return `(x, kv)` so `Llama3Model` can build `new_kv_cache`.
4. **`GroupQueryAttention.forward` variable mismatch.** After concatenating `past_kv`, the code references `k` and `v` variables that are undefined — should use `keys` and `values` or assign `k = keys` after expansion.
5. **`cos` and `sin` register_buffer:** they are saved with `persistent=False`, meaning they won't be saved into `state_dict()` by default. If you want them saved, use `persistent=True`.
6. **Top-k thresholding:** implementations use `torch.topk(logits, top_k)` but then incorrectly extract the threshold. Use:
```py
topk_vals, _ = torch.topk(logits, top_k)
min_vals = topk_vals[:, -1].unsqueeze(-1)
logits = torch.where(logits < min_vals, torch.tensor(-float('inf'), device=logits.device), logits)
```
7. **Device/dtype mismatches when assigning parameters:** always call `.to(device=device, dtype=...)` where appropriate.

### `siglip.py` issues
1. In `SiglipAttention.forward`, projections use `.view` incorrectly:
   - Fix: call projection modules: `self.k_proj(hidden_states)`.
2. `SiglipMLP.intermediate_size` default is suspiciously small (12) — set to `hidden_size * 4`.
3. `SiglipVisionTransformer.forward` incorrectly sets `hidden_states = self.embedding` instead of calling `self.embedding(pixel_values)`. Also `self.encoder` is called with wrong args.

### `processing_siglip.py` issues
1. `IMAGE_TOKEN` is local but `self.IMAGE_TOKEN` used — unify to `self.IMAGE_TOKEN = "<image>"`.
2. `tokens_to_add` key is `"additional_special tokens"` (space) — correct to `"additional_special_tokens"`.
3. `tokenizer.add_bos_token` toggles may not exist for the tokenizer API being used. For tiktoken, you may need to manage BOS/EOS behavior manually.

---

# Testing Checklist & Unit Tests to Add

Add pytest unit tests for:
- `compute_rope_params` returns cos/sin with expected shapes and values for toy `head_dim=8`, `context_length=16`.
- `rope` preserves norms and shapes.
- `GroupQueryAttention` returns correct output shape `(B, seq_len, emb_dim)` and attention weights sum to 1 across last axis.
- `SiglipAttention` attention weights shape verification and numeric stability with random inputs.
- `Tokenizer.encode/decode` roundtrip tests for sample texts and special tokens.
- `load_weights_into_llama` mapping with toy parameters (random tensors with expected shapes) to ensure no shape mismatch.

---

# Performance, Memory & Precision guidance

- Default `dtype=torch.bfloat16` reduces memory. However, consumer GPUs (e.g., GTX 1650) do not support bfloat16 — use `torch.float16` if CUDA and your PyTorch build support it; otherwise fallback to `torch.float32`.
- For generation with long contexts, use the KV‑cache implementation to avoid recomputing past key/values and to reduce overall compute.
- Use `torch.cuda.amp.autocast()` and `GradScaler` during training to reduce memory and speed up mixed‑precision training.
- If you run into OOM, options:
  - Reduce `batch_size` or use gradient accumulation.
  - Lower model `dtype`.
  - Use smaller `LoRA.rank`.
  - Offload parameters to CPU (slow) or use ZeRO (requires DeepSpeed or FairScale).
- Memory estimate per layer roughly: keys + values for GQA ~ `batch * num_kv_groups * seq_len * head_dim * 2 * 4 bytes` (float32) — tune accordingly.

---

# Contributing, License & Citation

**Contributing**
- Fork the repo, add tests, and create PRs.
- Suggested PR tags: `bugfix/rope`, `feature/lora-wiring`, `feature/multimodal-fusion`.

**License**
- Add a `LICENSE` (MIT or Apache‑2.0 recommended). Note: **model weights** are under Meta’s LLaMA license and require HF gating/acceptance.

**Citation**
```bibtex
@software{llama32_siglip_multimodal_2025,
  title  = {LLaMA‑3.2 Multimodal (WIP)},
  year   = {2025},
  note   = {Text backbone + SigLIP Vision + preprocessing},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```

---

# Changelog — what's new (from your uploaded files)
This README incorporates and expands content from the original README plus the three uploaded source files:

- **`model.py`**: added full code reference for LLaMA‑3.2 config, tokenizer wrapper, RoPE (frequency scaling), Grouped Query Attention (GQA) with KV‑cache, `FeedForward`, `RMSNorm`, two `generate` variants, LoRA class `Linear_LORA`, `load_weights_into_llama`, and a training script skeleton (HF hub downloads).
- **`processing_siglip.py`**: image preprocessing utilities (resize, rescale, normalize, process_images) and `paligemmaProcessor` (adds image tokens to prompts and returns pixel values + tokenizer inputs). README documents the default mean/std/rescale and corrections needed for tokenizer integration.
- **`siglip.py`**: SigLIP vision backbone (patch conv embedding, position embeddings, `SiglipAttention`, MLP, encoder, and `SiglipModel`). README points out code fixes needed (projection calls, transformer forward).

---

# Where to go next (practical TODOs)
- Fix the small but critical bugs called out above in `siglip.py` and `processing_siglip.py`.
- Decide on your multimodal fusion approach — start with **prepend visual tokens** and the `AdaptiveAvgPool1d` projection trick.
- Wire `Linear_LORA` to attention and MLP projections (create a helper `convert_to_lora(model, target_modules, rank, alpha)`).
- Add unit tests for all numerical modules (`rope`, `GQA`, `SiglipAttention`).
- Add example notebooks demonstrating:
  - Text-only generation (1B size).
  - Visual token prepending demo with one image and a question (toy).
  - Saving/loading LoRA adapters and applying them.

---

# Appendix — Short Code Snippets & Utilities

## Convert existing `nn.Linear` to `Linear_LORA` (helper)
```py
def replace_linear_with_lora(module, rank=8, alpha=32, dropout=0.0, target_module_names=None):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and (target_module_names is None or name in target_module_names):
            in_dim = child.in_features
            out_dim = child.out_features
            lora = Linear_LORA(in_dim, out_dim, rank, alpha, dropout)
            # Copy weights
            with torch.no_grad():
                lora.linear.weight.copy_(child.weight)
            setattr(module, name, lora)
        else:
            replace_linear_with_lora(child, rank, alpha, dropout, target_module_names)
```

## Example: forward helper accepting `inputs_embeds` in `Llama3Model`
```py
def forward_with_embeds(self, inputs_embeds, past_kv=None, use_cache=False):
    # Accepts (B, seq_len, emb_dim) directly
    x = inputs_embeds
    # reuse rest of forward logic but skip tok_emb lookup
    ...
```

---

If you'd like, I can:
- Produce a small unit test file (`tests/test_rope.py`) to exercise `compute_rope_params` + `rope`.  
- Create a starter script to convert an existing `Llama3Model` to LoRA‑adapted form.  
- Add a polished `examples/multimodal_demo.py` that wires SigLIP → projection → LLaMA and runs a single forward pass (best‑effort given API mismatches).

---

# Endnotes
This README is intentionally thorough to guide development, debugging, and experiments. It documents how the currently uploaded files map to the model, lists concrete implementation bugs found and how to fix them, and gives hands‑on examples for multimodal fusion and LoRA fine‑tuning.

---

