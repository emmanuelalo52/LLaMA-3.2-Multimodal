# 🦙 LLaMA‑3.2 Multimodal (WIP) — Text Backbone + LoRA + RoPE + GQA

> **Status:** 🚧 **Under active construction**. The **text generation backbone** is working (tokenizer, model blocks, RoPE, GQA attention, KV‑cache, weight loading, and basic generation). **Multimodal** components (vision/audio encoders and fusion) are planned and partly scaffolded but **not wired up yet**.

---

## TL;DR
- ✅ **Implemented now:** LLaMA‑3.2 config, Tokenizer (tiktoken‑BPE wrapper), Rotary Positional Embedding (RoPE) with frequency scaling, **Grouped‑Query Attention (GQA)**, **RMSNorm**, FeedForward (SiLU/SwiGLU‑style), **KV‑cache**, **LoRA** module (not yet plugged into all projections), **safetensors** weight loading, HF Hub download helpers, and a **sampling `generate`** loop.
- 🚧 **In progress:** LoRA wiring into attention/MLP projections; training harness; proper unit tests; benchmarks.
- 🧠 **Planned multimodality:** Vision encoder (CLIP/ViT‑like), optional audio encoder, **modality fusion** (projection or cross‑attention), alignment/fine‑tuning datasets, and safety filters.

---

## What this repo aims to be
A compact, readable **LLaMA‑3.2–style multimodal stack** you can extend:
- Start with the provided **text‑only backbone** (already functional).
- **Add vision/audio** front‑ends that map non‑text inputs into the model’s token/embedding space.
- Experiment with **fusion** strategies (learned projections, gated cross‑attention, or lightweight Perceiver‑style adapters).
- Fine‑tune end‑to‑end with **LoRA/PEFT** to keep training efficient.

---

## Implemented Components (Now Working)

### 1) Configuration — `LLAMA32_CONFIG`
- `vocab_size=128_256`, `context_length=131_072`, `emb_dim=2048`, `n_heads=32`, `n_layers=16`
- GQA via `n_kv_groups=8` (keys/values shared across head groups)
- Long‑context **RoPE** settings with frequency scaling
- Default `dtype=torch.bfloat16` (reduce memory; GPU recommended)

### 2) Tokenizer — `Tokenizer`
- Thin wrapper around **tiktoken** with **LLaMA‑3.2 special IDs**:
  - `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` and reserved range
- `encode(text, bos=False, eos=False)` and `decode(ids)` helpers

### 3) Core Blocks
- **RMSNorm** normalization (`RMSNorm(dim, eps)`)
- **FeedForward** (SwiGLU‑style: `silu(W1 x) ⊙ W2 x → W3`)
- **RoPE** utilities: `compute_rope_params(...)` + `rope(...)` (2‑D rotary embedding)
- **GQA Attention** with optional **KV‑cache** for fast autoregressive inference
- **TransformerBlock** combining attention + MLP, residual paths
- **Llama3Model**: embeddings → N transformer blocks → final norm → LM head

### 4) LoRA — `Linear_LORA`
- Minimal low‑rank adapter with frozen base linear
- Trainable `lora_a` and `lora_b` plus dropout; scales by `alpha/rank`
- 🧩 **Wiring note:** The class is implemented; replacement hooks into attention/MLP projections are **TBD** (see Roadmap).

### 5) Weights & Generation
- **Load** pretrained weights from **Hugging Face** (`meta-llama/Llama-3.2-*-Instruct`) via `safetensors`
- `load_weights_into_llama(model, cfg, params)` assigns tensors
- **Sampler**: `generate(...)` supports **temperature** and **top‑k** filtering; uses **KV‑cache**

---

## Multimodal Scope (Planned)

- **Vision**: ViT/CLIP‑style image encoder → projection into text embedding space → prepend/insert as “visual tokens”
- **Audio** (optional): Whisper‑like encoder or spectrogram CNN → projection → tokens
- **Fusion**: 
  - *Lightweight:* linear projection + position tags  
  - *Richer:* **cross‑attention** layers gated by learned prompts  
  - *PEFT‑ready:* LoRA adapters on fusion layers for efficient fine‑tuning
- **Training Phases**: modality alignment → instruction tuning → task‑specific adapters
- **Datasets**: LLaVA‑style image‑caption, VQA, multimodal instruction sets (respect licenses and usage policies)

---

## Getting Started

### 1) Install
```bash
# (optional) venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch safetensors tiktoken huggingface_hub
```

> PyTorch build should match your CUDA if you want GPU acceleration.

### 2) Hugging Face access
LLaMA‑3.2 weights require accepting Meta’s license on HF and being granted access.
```python
from huggingface_hub import login
login()  # or login(token="hf_xxx")
```

### 3) Download tokenizer + weights (example)
```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

LLAMA_SIZE_STR = "1B"  # e.g. "1B", "3B", "8B"

tok_path = hf_hub_download(
    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    filename="original/tokenizer.model",
    local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
)

# For 1B single-shard:
weights_path = hf_hub_download(
    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    filename="model.safetensors",
    local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
)
params = load_file(weights_path)  # dict[str, Tensor]
```

### 4) Build model, load weights, generate
```python
import torch
from pathlib import Path

# Assuming the classes/functions live in your module file
# from llama32_multimodal import (
#     LLAMA32_CONFIG, Tokenizer, Llama3Model,
#     load_weights_into_llama, generate
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Llama3Model(LLAMA32_CONFIG).to(device)

# Load tensors
load_weights_into_llama(model, LLAMA32_CONFIG, params)
model.eval()

# Tokenizer
tokenizer = Tokenizer(tok_path)

# Encode prompt and run
prompt = "### Instruction:\nWrite a motivational line about persistence.\n\n### Response:\n"
ids = tokenizer.encode(prompt, bos=True, eos=False)
idx = torch.tensor(ids, device=device).unsqueeze(0)  # (1, seq_len)

with torch.no_grad():
    out = generate(
        model, idx, max_new_tokens=64,
        context_size=LLAMA32_CONFIG["context_length"],
        temperature=0.8, top_k=50
    )

print(tokenizer.decode(out.squeeze(0).tolist()))
```

---

## Repository Structure (suggested)
```
.
├── llama32_multimodal.py      # This module (model blocks, tokenizer, LoRA, RoPE, GQA, loading, generate)
├── examples/
│   └── text_inference.py      # Minimal example to run generation (uses HF hub + safetensors)
├── data/                      # (optional) sample prompts, test assets
└── README.md
```

---

## Roadmap

### Short‑term (text backbone)
- [ ] Wire **LoRA** into attention `q/k/v/o` and MLP `up/gate/down` projections with toggleable replacement utility
- [ ] Unit tests for RoPE, KV‑cache, masking, and shape invariants
- [ ] CLI flags for model size, decoding params, and dtype
- [ ] Save/load adapter weights

### Multimodal
- [ ] **Vision encoder** (ViT/CLIP‑style) + projection to text space
- [ ] (Optional) **Audio encoder** + projection
- [ ] **Fusion layers** (cross‑attention + learned prompts)
- [ ] Instruction‑tuning datasets and training loop
- [ ] Safety filters & evals (toxicity, jailbreak probes)

---

## Known Issues / Fix‑Ups To Do
The backbone is a **work‑in‑progress**; keep these items in mind if you extend the code:
- `compute_rope_params(...)`: variable naming (`inv_freq`) and arg names (`theta` vs `theta_base`) should be consistent.
- `rope(...)`: ensure even `head_dim`; validate shapes on cat/rotate.
- `FeedForward`: constructor currently uses `dtype=["dtype"]` — replace with the actual dtype (e.g., `dtype=cfg["dtype"]`) and fix in/out dims (`fc2` should map `emb_dim→hidden_dim` or `hidden_dim` consistently).
- `GroupQueryAttention.forward(...)`: keep naming consistent (`keys/values` vs `k/v` when caching) and return both `(context, kv)` if `use_cache=True`; verify mask dtype is `bool`.
- `TransformerBlock.forward(...)`: should **return** `(x, kv)` when caching is enabled; residual connections should use distinct shortcuts for attn/ffn.
- `Llama3Model`: `RMSNorm` signature vs usage should match; confirm `compute_rope_params` call args (e.g., `theta`).
- Two `generate(...)` functions exist in the code sample — pick the KV‑cache one and remove duplicates.
- `top_k` filtering: `torch.topk` returns `(values, indices)` — use the **values** as threshold and preserve device/dtype on the `-inf` tensor.
- Always move tensors to the correct **device/dtype** before assignment (`assign(...)`).

> PRs welcome! If you want, open an issue and I’ll help triage any shape/dtype questions.

---

## Design Notes
- **Why GQA?** Cuts KV memory by sharing keys/values across head groups while keeping many query heads → **faster inference** and **lower memory**.
- **Why RoPE scaling?** Extends usable context windows beyond training length with minimal changes.
- **Why LoRA/PEFT?** Enables task‑specific tuning without touching the full parameter set; crucial for multimodal alignment.

---

## Licensing & Use
- **Model weights**: subject to **Meta LLaMA 3.2** license and gating on Hugging Face. You must request/accept access for each size you use.
- **This code**: add your own license file (MIT/Apache‑2.0 suggested). If you copy code from other repos, preserve their attributions.
- **Datasets**: respect licenses/terms and handle sensitive content responsibly.

---

## Citation
If this codebase helps your research or product demos:
```
@software{llama32_multimodal_wip_2025,
  title  = {LLaMA-3.2 Multimodal (WIP)},
  year   = {2025},
  note   = {Text backbone with RoPE, GQA, KV-cache, LoRA scaffolding},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```

---

### Thanks
Big thanks to the open‑source community around **LLaMA**, **tiktoken**, **safetensors**, and **PEFT/LoRA** for the building blocks that make this possible.
