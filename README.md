
# MLLAMA using LLAMA 3.2 VLM configuration and Siglip

> A thorough, practical, and implementation-focused README for the multimodal codebase including the Siglip vision transformer, Llama-style language model, and the image processing utilities. This repository explains APIs, gives examples, and suggests training/debugging guidance.

---
## Table of contents
1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Installation & dependencies](#installation--dependencies)
4. [Quickstart examples (run locally)](#quickstart-examples-run-locally)
5. [API reference — vision (Siglip) components](#api-reference---vision-siglip-components)
6. [API reference — language & multimodal model (model.py)](#api-reference---language--multimodal-model-modelpy)
7. [Image processing (processing_mllama.py)](#image-processing-processing_mllamapy)
8. [Testing, debugging & utilities](#testing-debugging--utilities)
9. [Training guidance, tips & best practices](#training-guidance-tips--best-practices)
10. [Extending the codebase (LoRA, KV cache, export)](#extending-the-codebase-lora-kv-cache-export)
11. [Changelog / Known limitations](#changelog--known-limitations)

---
## Overview

This repository implements a small vision transformer (Siglip) and a Llama-style language model and glues them together into a multimodal model (MllamaForConditionalGeneration). The components are intentionally modular:
- A Vision encoder (Siglip) that converts images into patch embeddings.
- A projection layer that maps vision embeddings into the language model embedding space.
- A Llama-like autoregressive transformer that consumes merged text+image embeddings for multimodal generation.

---
## Repository structure (what I saw)

- `siglip.py` — **(the Siglip vision transformer code you pasted in chat)** — Vision encoder implementation (patch embedding, position embedding, multi-head attention, MLP and transformer encoder layers). 
- `model.py` — Language + multimodal model (Llama variants, rotary embeddings, grouped-query attention, multimodal projector, and the top-level `MllamaForConditionalGeneration`). fileciteturn0file0
- `processing_mllama.py` — Image preprocessing and `MllamaImageProcessor` helper (resizing, normalization, token insertion). fileciteturn0file1

---
## Installation & dependencies

Recommended environment:
- Python 3.9+ (3.10 recommended)
- PyTorch (1.13+ recommended; for best float16 support use a recent stable build matching CUDA)
- torchvision (optional, for datasets)
- numpy, pillow
- safetensors (only if you will load safetensors weights)
- tiktoken (optional if you plan to use the tokenizer pieces this repo references)
- (optional) HuggingFace `transformers` and `datasets` if you integrate HF tokenizers/datasets

Example `pip` install:
```bash
python -m pip install torch torchvision numpy pillow safetensors tiktoken
# optionally:
python -m pip install transformers datasets
```

GPU & dtype note:
- `LLAMA32Config` defaults to `torch.bfloat16` in the code but most consumer NVIDIA GPUs (e.g., GTX/RTX) prefer `torch.float16`. Set dtype accordingly (`config.dtype = torch.float16`) if you use CUDA compute capability that doesn't support bfloat16 efficiently.
- Mixed precision (AMP) is recommended for training large models (use `torch.cuda.amp.autocast` + `GradScaler`).

---
## Quickstart examples (run locally)

Below are small runnable snippets showing how to instantiate components and run a forward pass with dummy data. These assume the files are in the Python path and named `siglip.py`, `model.py`, `processing_mllama.py`.

### 1) Minimal vision forward (Siglip)
```python
import torch
from siglip import SiglipVisionConfig, SiglipModel

# configure and create
vis_cfg = SiglipVisionConfig(hidden_size=768, image_size=224, patch_size=16, num_hidden_layers=2, num_attention_heads=8)
vis_model = SiglipModel(vis_cfg).eval()

# dummy image batch: [B, C, H, W]
x = torch.randn(2, 3, vis_cfg.image_size, vis_cfg.image_size)
features = vis_model(x)  # -> [B, num_patches, hidden_size]
print("Vision features:", features.shape)
```

### 2) Minimal multimodal forward (Mllama)
```python
import torch
from model import MLLAMAConfig, LLAMA32Config, MllamaForConditionalGeneration, MultiModalProjector
from siglip import SiglipVisionConfig, SiglipModel

# build configs (example shapes)
vision_cfg = {"hidden_size": 768, "image_size": 224, "patch_size": 16, "num_hidden_layers": 2, "num_attention_heads": 8}
text_cfg = {"vocab_size": 128256, "hidden_size": 2048, "n_heads": 16, "n_layers": 4, "head_dim": 128, "hidden_dim": 8192}

m_cfg = MLLAMAConfig(vision_config=vision_cfg, text_config=text_cfg, projection_dim=2048, vocab_size=128256)
model = MllamaForConditionalGeneration(m_cfg)

# dummy inputs
batch = 1
seq_len = 64
input_ids = torch.randint(0, m_cfg.vocab_size, (batch, seq_len))
pixel_values = torch.randn(batch, 3, m_cfg.vision_config.image_size, m_cfg.vision_config.image_size)

out = model(input_ids=input_ids, pixel_values=pixel_values)
print("Logits:", out["logits"].shape)  # -> [batch, seq_len, vocab_size]
```

> Note: these minimal examples are for smoke-testing shapes; see "Common issues" below for compatibility fixes that you'll likely need to apply before everything runs.

---
## API reference — vision (Siglip) components

The Siglip components implement a small patch-based vision transformer. Below are the important classes and their behavior (constructor args & shapes).

### `SiglipVisionConfig`
- Args (defaults shown in code):
  - `hidden_size=768`
  - `intermediate_size=3072`
  - `num_hidden_layers=12`
  - `num_attention_heads=12`
  - `num_channels=3`
  - `image_size=224`
  - `patch_size=16`
  - `layer_norm_eps=1e-6`
  - `attention_dropout=0.0`
  - `num_image_tokens: int = None`
- Purpose: centralizes model hyperparameters for the Siglip vision stack.

### `SiglipVisionEmbedding(nn.Module)`
- Responsibilities:
  - Splits image into non-overlapping patches with a `nn.Conv2d` (kernel_size=patch_size, stride=patch_size).
  - Produces patch embeddings: `[batch, num_patches, embed_dim]`.
  - Adds positional embeddings (registered buffer `position_ids` used to index `nn.Embedding`).
- Forward:
  - Input: `pixel_values` tensor shaped `[B, C, H, W]` (C should equal `num_channels`).
  - Output: `[B, num_patches, hidden_size]`

**Important shape computations:**
- `num_patches = (image_size // patch_size) ** 2`
- The patch convolution yields `[B, embed_dim, num_patches_h, num_patches_w]` and the code flattens to `[B, embed_dim, num_patches]` then transposes to `[B, num_patches, embed_dim]`.

### `SiglipAttention` (multi-head)
- Inputs: `hidden_states` `[B, seq_len, embed_dim]`
- Internals:
  - Linear projections for query/key/value: `q_proj`, `k_proj`, `v_proj` (all project to `embed_dim`).
  - Reshapes to `[B, num_heads, seq_len, head_dim]` for attention computation.
  - Computes scaled dot-product attention and returns `(context_vector, attention_weights)`.
- Output: `context_vector` `[B, seq_len, embed_dim]`, `attention_weights` `[B, num_heads, seq_len, seq_len]`.

### `SiglipMLP`
- Simple two-layer feed-forward module with GELU activation (approximate variant used in the code). Input: `[B, seq_len, embed_dim]` and output `[B, seq_len, embed_dim]`.

### `SiglipVisionEncoder` and `SiglipEncoder`
- `SiglipVisionEncoder` implements a single transformer encoder block with:
  - `LayerNorm` -> `SelfAttention` -> residual add -> `LayerNorm` -> `MLP` -> residual add.
- `SiglipEncoder` stacks `num_hidden_layers` of these blocks (`nn.ModuleList`).

### `SiglipVisionTransformer` and `SiglipModel`
- `SiglipVisionTransformer` composes `SiglipVisionEmbedding` + `SiglipEncoder` + final `LayerNorm` and returns `last_hidden_state` of shape `[B, num_patches, hidden_size]`.
- `SiglipModel` is a tiny wrapper that instantiates the transformer and provides a `forward(pixel_values)` that returns the transformer outputs.

---
## API reference — language & multimodal model (`model.py`)

`model.py` contains a Llama-style autoregressive model implementation, rotary embeddings, grouped-query attention (GQA), and the multimodal wrapper.

Main components documented here (shortened names):

### `LLAMA32Config`
- Holds architecture & runtime settings for the Llama-like model (hidden size, n_heads, n_layers, dtype, etc.).
- **Important**: `dtype` defaults to `torch.bfloat16` in the file. Set this to `torch.float16` for float16 on GPUs that don't support bfloat16.

### `MLLAMAConfig`
- Top-level multimodal config (holds `vision_config`, `text_config`, `projection_dim`, `image_token_index`, `vocab_size`).
- Internally wraps `vision_config` into `SiglipVisionConfig` and `text_config` into `LLAMA32Config`.

### `Linear_LORA`
- A convenience module that wraps a frozen linear layer plus low-rank LoRA adapters (`lora_a` and `lora_b`).
- Usage: replace some `nn.Linear` modules with `Linear_LORA(in_dim,out_dim,rank,alpha,dropout)` during fine-tuning to enable LoRA training.

### `LLAMARMSNorm`
- RMS normalization used by LLaMA variants (weights are trainable parameters).

### Rotary embeddings & helpers
- `LLAMARotaryEmbedding` computes cos/sin frequency tensors for RoPE.
- `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)` applies rotation to query/key tensors.
- `rotate_half(x)` helper for rotation math.

### `GroupQueryAttention` (GQA)
- Implements grouped-query attention with KV-groups and optional KV cache integration.
- Key functions: query/key/value projections, RoPE application, `repeat_kv` helper (maps kv groups to attention heads), attention score computation, optional addition of causal mask, softmax normalization, and out projection.
- Input shapes:
  - `hidden_states` `[B, seq_len, hidden_size]`
  - Produces `context_vector` shaped `[B, seq_len, hidden_size]`

### `TransformerBlock` & `Llama3Model` & `Llama3ForCausalLM`
- `TransformerBlock` wraps the attention, RMS norms, and feed-forward network.
- `Llama3Model` stacks `n_layers` transformer blocks and applies a final RMSNorm.
- `Llama3ForCausalLM` applies a final linear head (`lm_head`) mapping hidden states to vocabulary logits.

### `MultiModalProjector`
- Linear projection from vision hidden size into `projection_dim` (language embedding dimension). Use to map `Siglip` outputs into LM input space.

### `MllamaForConditionalGeneration`
- Top-level multimodal class that composes `SiglipModel` (vision), `MultiModalProjector`, and `Llama3ForCausalLM` (language).
- `forward(...)` accepts:
  - `input_ids` (text token ids) or `input_embeds` (already embedded tokens)
  - `pixel_values` (images as float tensors `[B, C, H, W]`)
  - `attention_mask`, `position_ids`, `image_mask`, `labels`, `kv_cache`, etc.
- Behavior:
  1. If `pixel_values` provided -> extract vision features via `self.vision_model` -> project via `self.multi_modal_projector` to `[B, num_image_patches, projection_dim]`.
  2. If `input_ids` provided, get `inputs_embeds` via language model's token embedding.
  3. If both present, `_merge_input_ids_with_image_features` merges by finding `image_token_index` tokens in `input_ids` and replacing them with actual image features (patch embeddings) inside `inputs_embeds`.
  4. Forward through language model (text+image embeddings) to generate `logits`. If `labels` provided, a simple cross-entropy loss is computed (shifted targets).

**Important**: The current `forward()` uses a simple cross-entropy on shifted tokens when `labels` are given. There are helper functions `_compute_loss_with_masking` and `_create_conversation_mask` that provide masked loss semantics for VLM training — consider enabling them during multi-modal fine-tuning to avoid penalizing image tokens or system prompt tokens.

---
## Image processing (`processing_mllama.py`)

This module contains helper functions to preprocess images for the Siglip vision model and the `MllamaImageProcessor` class which prepares both the pixel tensors and the tokenized text inputs.

Key functions & their behavior:
- `rescale(image, scale, dtype=np.float32)` — multiplies pixel values by `scale` and casts to dtype.
- `normalize(image, mean, std)` — channel-wise normalization using provided mean/std.
- `resize(image, size, resample, reducing_gap)` — wrapper around PIL resize.
- `process_images(images, size, resample, rescale_factor, image_mean, image_std)` — pipeline that resizes, converts to `np.array`, rescales to `[0..1]` (if asked), normalizes, and rearranges to `[C, H, W]`, returning a list of these arrays.
- `MllamaImageProcessor` — tokenizes the prompt and returns a dict containing tokenized `input_ids`, `attention_mask`, etc. and `pixel_values` as a torch tensor.

**Important notes / pitfalls (see bug fixes below)**:
- The code uses `self.IMAGE_TOKEN` before setting it as `self.IMAGE_TOKEN` — this will raise a `NameError` or unexpected behavior. Also the `tokens_to_add` dictionary key should be `'additional_special_tokens'` (with underscores) when calling a HF tokenizer `add_special_tokens` method — the code uses `'additional_special tokens'` which is incorrect.
- The method returns `{"pixel Value": pixel_values, **inputs}` — the key name has a space and capital V; standard is `pixel_values`. This mismatch will cause accidental bugs when passing the returned dict into `MllamaForConditionalGeneration` which expects `pixel_values=` keyword.

---
## Testing, debugging & utilities

### Unit tests & smoke tests (recommended)
- Write a small test to check shapes for each module (embedding -> encoder -> model). Example:
```python
def test_siglip_shapes():
    cfg = SiglipVisionConfig(hidden_size=64, image_size=32, patch_size=8, num_hidden_layers=1, num_attention_heads=4)
    model = SiglipModel(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, (cfg.image_size // cfg.patch_size)**2, cfg.hidden_size)
```
- Test attention shapes and ensure attention weights shape is `[B, num_heads, seq_len, seq_len]`.

### Debugging tips
- Add `print` or `assert` statements inside attention and reshape operations to quickly trap shape mismatches.
- When you see `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, print the shapes of the tensors involved before the matmul.
- For dtype errors, ensure embeddings, positional embeddings, and rotary cos/sin tensors are of the same dtype and device before arithmetic.

---
## Training guidance, tips & best practices

- **Start small**: train on small subset / shorter sequences and ensure forward/backward works.
- **Freeze vision or language initially**: freeze the large part and train projector + head first to stabilize training.
- **Use LoRA (built-in `Linear_LORA`)**: swap heavy linear layers to LoRA adapters if you don’t want to fine-tune all weights.
- **Batch size & memory**:
  - Use gradient accumulation for larger effective batch sizes on limited GPU memory.
  - Use mixed precision (AMP) for memory savings and speed.
- **Learning rate**:
  - For LoRA adapters: typical LR in range `1e-4 - 1e-3` with AdamW.
  - For full fine-tuning: `1e-5 - 5e-5` depending on batch size and model size.
- **Loss masking for VLMs**: when training multimodal models, use `_compute_loss_with_masking` that masks out image tokens and system messages to avoid incorrect supervision on image tokens. Replace the simple CE in `forward()` with that masked loss logic.

---
## Extending the codebase (LoRA, KV cache, export)

- **LoRA**: `Linear_LORA` is available — to use it, replace `nn.Linear` instances in the language model with `Linear_LORA` wrappers for adapter-style fine-tuning. Ensure the original linear weights are set `requires_grad=False` as in the provided class, and LoRA parameters are left trainable.
- **KV cache**: `GroupQueryAttention` accepts a `kv_cache` parameter but you will need to provide a KV cache object implementing `.update(keys, values, layer_idx)` and returning updated `keys, values`. The current code expects KV caching protocol but does not ship a KV cache class.
- **Export**: After training, export weights to `safetensors` or standard `torch.save(model.state_dict())`. Remember to move buffers (rotary `inv_freq`) to CPU if saving for downstream frameworks that expect CPU tensors.

---
## Changelog / Known limitations
- `MllamaForConditionalGeneration.forward` currently computes simple shifted cross-entropy. For real VLM training, use `_compute_loss_with_masking` (or plug in dataset-specific masking logic).

---
