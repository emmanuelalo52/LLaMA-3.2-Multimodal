# VLM-LLaMA3 — Multimodal LLaMA-3.2 with Plain ViT Vision Encoder

A clean PyTorch implementation of a Vision-Language Model (VLM) combining:

- **Plain ViT** vision encoder (replaces SigLIP — no contrastive/sigmoid pair-wise loss)
- **LLaMA-3.2** causal language model (GQA, RoPE, RMSNorm)
- **Custom CUDA kernels** — fused SwiGLU (`Tools/swiglu/swiglu.cu`) and fused Add-RMSNorm (`Tools/rmsnorm/rmsnorm.cu`)

---

## What changed from the original repo

| Component | Before | After |
|---|---|---|
| Vision tower | SigLIP (`Model/siglip.py`) | Plain ViT (`Model/vision_encoder.py`) |
| Vision loss | Sigmoid pairwise contrastive | None (feature extractor only) |
| `pixel_values` key | `"pixel Value"` (with space) | `"pixel_values"` (fixed) |
| Weight download | Manual | `weights/download_weights.py` |
| `Model/__init__.py` | Exported SigLIP | Exports VisionEncoder |

---

## Repository structure

```
.
├── Model/
│   ├── vision_encoder.py    ← NEW: plain ViT (replaces siglip.py)
│   ├── model.py             ← updated: uses VisionEncoder
│   ├── processing_mllama.py ← updated: fixed pixel_values key
│   ├── utils.py             ← updated: weight key map for plain ViT
│   └── __init__.py
├── Tools/
│   ├── rmsnorm/
│   │   ├── rmsnorm.cu       ← fused Add-RMSNorm CUDA kernel (unchanged)
│   │   └── rmsnorm.cuh
│   └── swiglu/
│       ├── swiglu.cu        ← fused SwiGLU CUDA kernel (unchanged)
│       ├── swiglu.cuh
│       ├── swiglu_binding.cpp
│       ├── FusedSwiglu.py
│       └── __init__.py
├── Inference/
│   └── Inference.py         ← updated: supports custom + HF inference
├── weights/
│   ├── download_weights.py  ← NEW: downloads instruct weights from HF
│   └── README.md
└── setup.py                 ← builds rmsnorm + swiglu_fused CUDA extensions
```

---

## Quick start

### 1. Install dependencies

```bash
pip install torch torchvision safetensors transformers huggingface_hub pillow numpy
```

### 2. Build CUDA extensions

```bash
python setup.py build_ext --inplace
```

### 3. Download instruct weights

```bash
huggingface-cli login          # one-time — LLaMA is gated
python weights/download_weights.py
```

### 4. Run inference

```bash
python Inference/Inference.py \
    --image path/to/image.jpg \
    --prompt "Describe the image." \
    --hf-weights weights/Llama-3.2-11B-Vision-Instruct
```

---

## Vision Encoder (`Model/vision_encoder.py`)

### Architecture

```
pixel_values [B, 3, H, W]
       │
  ViTPatchEmbeddings     Conv2d(patch_size stride) + learned positional embedding
       │
  ViTEncoder             N × ViTEncoderBlock
       │                   Pre-norm: LayerNorm → MHA → residual
       │                   Pre-norm: LayerNorm → GELU MLP → residual
       │
  post_layernorm         Final LayerNorm
       │
  patch_embeds [B, num_patches, hidden_size]
```

### Config (`VisionEncoderConfig`)

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 1280 | ViT-H patch embedding dimension |
| `intermediate_size` | 5120 | FFN inner dimension |
| `num_hidden_layers` | 32 | Transformer blocks |
| `num_attention_heads` | 16 | MHA heads |
| `image_size` | 560 | Input resolution (px) |
| `patch_size` | 14 | Patch size (px) |

Defaults match **LLaMA-3.2-Vision** tile encoder dimensions.

---

## Programmatic usage

```python
import torch
from Model import MLLAMAConfig, MllamaForConditionalGeneration

vision_cfg = {
    "hidden_size": 1280, "intermediate_size": 5120,
    "num_hidden_layers": 32, "num_attention_heads": 16,
    "image_size": 560, "patch_size": 14,
}
text_cfg = {
    "vocab_size": 128256, "hidden_size": 4096,
    "n_heads": 32, "n_layers": 32,
    "hidden_dim": 14336, "n_kv_groups": 8,
}
cfg   = MLLAMAConfig(vision_config=vision_cfg, text_config=text_cfg, projection_dim=4096)
model = MllamaForConditionalGeneration(cfg)
model.eval()
```

### Load HF instruct weights

```python
from Model.utils import load_hf_model

model, tokenizer = load_hf_model(
    "weights/Llama-3.2-11B-Vision-Instruct",
    device="cuda",
)
```

---

## CUDA extensions

Both kernels are compiled by `setup.py` and used automatically at runtime when:
- A CUDA GPU is available
- Input dtype is `float16` or `bfloat16`

| Kernel | File | Used by |
|---|---|---|
| Fused Add-RMSNorm | `Tools/rmsnorm/rmsnorm.cu` | `LLAMARMSNorm` in `model.py` |
| Fused SwiGLU | `Tools/swiglu/swiglu.cu` | `FusedFeedforward` in `model.py` |

Pure-PyTorch fallbacks activate automatically on CPU or `float32`.

---

## License

MIT