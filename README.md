# MLLAMA — Multimodal LLaMA-style Vision+Language Model  
**A compact, research-friendly PyTorch implementation of a multimodal LLM** combining a ViT-like *Siglip* vision tower with a LLaMA-inspired causal language model and utilities for image-token merging, LoRA fine-tuning wrappers, and preprocessing.

---

> This README documents the code in this repository (key files: `model.py`, `processing_mllama.py`, `siglip.py`) and explains how to run inference, fine-tune with LoRA, and extend the codebase.

---

# Table of contents
1. [Project overview](#project-overview)  
2. [Repository structure](#repository-structure)  
3. [Key design & components](#key-design--components)  
4. [Requirements & installation](#requirements--installation)  
5. [Quick start — inference example](#quick-start---inference-example)  
6. [LoRA fine-tuning (concept + example)](#lora-fine-tuning-concept--example)  
7. [Data preprocessing and tokenization notes](#data-preprocessing-and-tokenization-notes)  
8. [Saving / loading weights & checkpoints](#saving--loading-weights--checkpoints)  
9. [API reference (brief)](#api-reference-brief)  
10. [Tips, caveats & TODOs / known issues](#tips-caveats--todostknown-issues)  
11. [Contributing, license & acknowledgements](#contributing-license--acknowledgements)

---

# Project overview
This repository implements a modular multimodal model that:

- Encodes images using a ViT-like **Siglip** vision transformer, returning patch embeddings.  
- Encodes text with a LLaMA-style transformer (Group-Query attention, rotary embeddings, RMSNorm) and supports combining image patch embeddings into the decoding stream.  
- Provides utilities for image preprocessing and special token handling for image placeholders.  
- Includes a LoRA-style lightweight adapter wrapper `Linear_LORA` to fine-tune with far fewer trainable parameters.  

The design focuses on clarity & extendability for research experiments (VLM training, LoRA fine-tuning, KV-cache inference).

---

# Repository structure
- `model.py` — Core multimodal model, LLaMA-like transformer blocks, rotary embeddings, Group-Query attention, LoRA wrapper, multimodal merging and loss helpers.  
- `siglip.py` — Vision transformer: embeddings, attention, MLP and encoder layers for extracting patch features.  
- `processing_mllama.py` — Image preprocessing utilities and `MllamaImageProcessor` for creating image placeholders in prompts and returning tensors suitable for the model. **Note:** returns a key named `"pixel Value"` (contains space) — see caveats.  
- (Optional) other utility scripts / notebooks (not included here): training loop, dataset loader, HF `Trainer` wrappers.

---

# Key design & components (conceptual)
### Vision tower — Siglip
- Patch embedding via a Conv2d with `patch_size` stride → positional embeddings → transformer encoder stack. Output shape: `[batch, num_patches, embed_dim]`.

### Language tower — LLaMA-like
- Embedding layer, stack of `TransformerBlock` instances, RMS-style layer norm and linear LM head. Includes:
  - **GroupQueryAttention** (grouped key/value heads → repeated for attention)  
  - **RoPE** (rotary positional embeddings)  
  - **KV cache support** to enable efficient autoregressive generation.

### Multimodal merging
- Image features (vision patch embeddings) are projected with a `MultiModalProjector` then **merged into text token embeddings** by replacing a sequence of special image placeholder tokens (e.g. `<image>`) with the actual patch embeddings at the first image token position in the sequence. This is implemented in `_merge_input_ids_with_image_features`.

### LoRA wrapper
- `Linear_LORA` wraps an `nn.Linear` with frozen main weights and trainable low-rank adapters `lora_a` and `lora_b`, enabling efficient fine-tuning. Use by replacing select linear layers in the LM.

---

# Requirements & installation

**Minimum / recommended packages**
```bash
pip install torch torchvision      # PyTorch (CPU/GPU build appropriate for your system)
pip install safetensors            # safe checkpoint format helpers
pip install tiktoken               # tokenizers utilities if needed
pip install numpy pillow           # preprocessing
pip install transformers           # optional — for a tokenizer (recommended)
```

Create a `requirements.txt` if you prefer:
```
torch
safetensors
tiktoken
numpy
pillow
transformers
```

> GPU tip: this code uses lower-precision dtypes (e.g. `torch.float16`) in several modules by default — running on a CUDA-enabled GPU with sufficient memory (adjust dtype and batch size for a GTX 1650) is recommended for reasonable throughput.

---

# Quick start — inference example

Below is a minimal example to run a forward pass. This is a **starting point** — you'll likely need to adapt tokenizers, vocabulary size, and checkpoint loading for your model weights.

```python
# quick_infer.py (example)
import torch
from PIL import Image

# import repo modules
from siglip import SiglipVisionConfig  # siglip.py
from model import MLLAMAConfig, MllamaForConditionalGeneration  # model.py
from processing_mllama import MllamaImageProcessor  # processing_mllama.py
# You will need a tokenizer — example uses HuggingFace LLaMA tokenizer
from transformers import LlamaTokenizer

# 1) Build configs (dicts -> MLLAMAConfig handles conversion internally)
vision_cfg = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
}
text_cfg = {
    "vocab_size": 128256,
    "hidden_size": 4096,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "max_position_embeddings": 2048,
    "context_length": 131072
}

cfg = MLLAMAConfig(vision_config=vision_cfg, text_config=text_cfg, projection_dim=2048)
model = MllamaForConditionalGeneration(cfg)  # uninitialized weights — load checkpoints for real usage
model.eval()

# 2) Tokenizer / image processor
tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-7b", use_fast=True)  # adapt to available tokenizer
# NOTE: MLLAMAConfig sets text_config.num_image_tokens = (image_size // patch_size) ** 2
num_image_tokens = cfg.text_config.num_image_tokens
image_processor = MllamaImageProcessor(tokenizer, num_image_token=num_image_tokens, image_size=cfg.vision_config.image_size)

# 3) Prepare input (single image + prompt)
img = Image.open("examples/cat.jpg").convert("RGB")
prompt = "Describe the image."

# processing_mllama returns a dict with keys like: 'pixel Value', 'input_ids', 'attention_mask' (note space)
batch = image_processor([prompt], [img], padding=True)
pixel_values = batch["pixel Value"]   # <-- watch the key name
input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]

# 4) Forward
with torch.no_grad():
    out = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

logits = out["logits"]      # shape: [batch, seq_len, vocab_size]
pred_ids = logits.argmax(-1)  # greedy decode (example)
decoded = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
print(decoded)
```

Notes:
- The example above creates an in-memory model. You should load pretrained weights (safetensors / state_dict) before real inference.  
- `processing_mllama.MllamaImageProcessor` returns the pixel data under the key `"pixel Value"` (space between words). That is intentional in the current code; you may rename it to `pixel_values` in the function for cleanliness.

---

# LoRA fine-tuning (concept + example)
**Concept:** wrap specific `nn.Linear` layers using the provided `Linear_LORA` wrapper to freeze base weights and train only low-rank adapters.

Simple utility to replace `nn.Linear` layers in your LM (example — adapt to your naming and layers you want to adapt):

```python
from model import Linear_LORA

def convert_linears_to_lora(module, rank=4, alpha=16, dropout=0.0, target_module_classes=(torch.nn.Linear,)):
    """
    Recursively replace Linear modules with Linear_LORA(wrapper) where desired.
    This is a basic example — refine selection to only replace attention projection matrices or FFN layers.
    """
    for name, child in module.named_children():
        if isinstance(child, target_module_classes):
            in_dim, out_dim = child.in_features, child.out_features
            lora = Linear_LORA(in_dim, out_dim, rank=rank, alpha=alpha, dropout=dropout)
            # copy original weights into lora.linear
            lora.linear.weight.data.copy_(child.weight.data)
            # replace in parent
            setattr(module, name, lora)
        else:
            convert_linears_to_lora(child, rank, alpha, dropout, target_module_classes)
```

After conversion:
- Only `lora.lora_a` and `lora.lora_b` (and their biases if enabled) will require grads. The base `linear.weight` is frozen.  

Training tips:
- Use `torch.cuda.amp` or `accelerate` mixed-precision.  
- Use small learning rates for LoRA adapters (e.g., 1e-4 – 1e-3) and a small batch size if GPU memory is limited.  
- Save only adapter parameters if you want a small checkpoint.

---

# Data preprocessing and tokenization notes
- `processing_mllama.py` defines `IMAGENET_STANDARD_MEAN` and `IMAGENET_STANDARD_STD` and helper functions to resize / normalize images and produce patch tensors. Pixel range is scaled using `rescale_factor = 1/255.0`.  
- `MllamaImageProcessor`:
  - Adds a special `<image>` token to the tokenizer (`tokenizer.add_special_tokens`) and creates additional placeholder tokens used for object location/segmentation.  
  - It sets `tokenizer.add_bos_token = False` and `tokenizer.add_eos_token = False` — tokenizers should be checked for expected behavior.  
  - **Important:** the processor returns `pixel_values` under the key `"pixel Value"`. Either use that exact key or modify the source to return `"pixel_values"`.

- The model's image-token merging logic (`_merge_input_ids_with_image_features`) replaces the first occurrence of the image token with the sequence of patch embeddings. If your dataset contains multiple `<image>` tokens or images at different positions, validate merging logic and adjust accordingly.

---

# Saving / loading weights & checkpoints
- The repository imports `safetensors.torch.load_file` in `model.py`, suggesting support for safetensors checkpoints. Example pseudo-load:
```python
from safetensors.torch import load_file

state = load_file("checkpoint.safetensors")
model.load_state_dict(state, strict=False)
```
- Alternatively use `torch.save(model.state_dict(), "ckpt.pt")` and `model.load_state_dict(torch.load(...))`.

Always ensure the checkpoint vocabulary (tokenizer) and config (vocab_size, image token id, etc.) are compatible.

---

# API reference (brief)
> Each line references the file that implements it.

### `siglip.py` (Vision)  
- `SiglipVisionConfig` — config fields (hidden_size, image_size, patch_size, num_hidden_layers, etc.).  
- `SiglipVisionEmbeddings` — converts pixel tensor to patch embeddings + positional embeddings.  
- `SiglipVisionTransformer` → `SiglipModel` — top-level forward that returns `[batch, num_patches, embed_dim]`.

### `processing_mllama.py` (Preprocessing)  
- `MllamaImageProcessor(tokenizer, num_image_token, image_size)` — callable that accepts lists `text` and `images`, returning a dict with `pixel Value`, `input_ids`, `attention_mask`, etc.  
- Helpers: `process_images`, `resize`, `normalize`, `rescale`, `add_image_tokens_to_prompts`.

### `model.py` (Core multimodal model)  
- `LLAMA32Config`, `MLLAMAConfig` — configuration classes. Note `MLLAMAConfig` converts nested `vision_config` → `SiglipVisionConfig` internally and populates `text_config.num_image_tokens`.  
- `Linear_LORA(in_dim, out_dim, rank, alpha, dropout)` — LoRA wrapper.  
- `LLAMARMSNorm`, `LLAMARotaryEmbedding`, `GroupQueryAttention`, `FeedForward`, `TransformerBlock` — low-level transformer components.  
- `MultiModalProjector` — projects vision embeddings to the LM hidden dim.  
- `Llama3Model` / `Llama3ForCausalLM` — LLM backbone and LM head.  
- `MllamaForConditionalGeneration` — high-level model gluing vision + language towers, handles image embedding insertion and computes loss if `labels` provided. Also contains helper masking functions for selective loss computation.  

---

# Tips, caveats & TODOs / known issues 
- **Tokenizer expectations** — Code relies on a tokenizer that can add special tokens and convert them to ids. `transformers` tokenizers (HF) are a good choice; ensure `vocab_size` matches the LM config.  
- **Loss masking** — `_compute_loss_with_masking` and `_create_conversation_mask` offer two approaches for selective loss computation; they assume your dataset uses `ignore_index` appropriately. Validate on your dataset.  
- **Checkpoint compatibility** — If you load checkpoints trained with different architectures (different `n_heads`, `hidden_size`, etc.), layers may mismatch. Use `strict=False` only when you understand missing/extra keys.  
- **KV cache & generation** — The model contains the structure for kv-cache updates in attention. Testing and integration with a generation loop will require implementing token-by-token generation loops that supply and update `kv_cache`.  

---

# Contributing
- Fork & create pull requests.  
- Add tests for: preprocessing (shapes & dtypes), forward pass (shapes for logits), LoRA replacements.  
- If you add a pretrained checkpoint, add a script that demonstrates loading it and running inference.

---

# License & citation
- **MIT**
---

# Acknowledgements
- Implementation inspired by standard transformer building blocks (rotary positional embeddings, ViT patch embeddings, Grouped-kv attention patterns) and LoRA techniques. See code-level comments in `model.py` and `siglip.py` for more contextual details.

---
