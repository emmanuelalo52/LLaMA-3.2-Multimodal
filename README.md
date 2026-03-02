# LLaMA-3.2 Multimodal

A research-grade PyTorch implementation of a Vision-Language Model built on LLaMA-3.2. It combines a Vision Transformer image encoder with a LLaMA-style causal language model, bridged by a single linear projector. Custom CUDA kernels accelerate the two most compute-intensive operations in the language model — RMSNorm and the SwiGLU feed-forward — with pure-PyTorch fallbacks for CPU or float32 runs.

---

## Architecture

The model has three components:

**Vision encoder** (`Model/vision_encoder.py`) — a standard pre-norm ViT that converts an image into a sequence of patch embeddings. It uses a Conv2d patch projection, learned absolute positional embeddings, multi-head self-attention with GELU MLP blocks, and a final LayerNorm. Output shape: `[B, num_patches, hidden_size]`.

**Multimodal projector** — a single linear layer that maps the vision encoder's hidden dimension into the language model's hidden dimension.

**Language model** (`Model/model.py`) — a LLaMA-3.2 decoder with Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), and fused SwiGLU feed-forward blocks. Supports KV caching for efficient autoregressive generation.

At inference time the patch embeddings are spliced directly into the token embedding sequence at the position of the `<image>` placeholder, then the language model decodes autoregressively from there.

---

## Repository structure

```
.
├── Model/
│   ├── vision_encoder.py      ViT image encoder
│   ├── model.py               LLaMA language model + full VLM
│   ├── processing_mllama.py   Image preprocessing and tokenisation
│   ├── utils.py               Weight loading from HF checkpoints
│   └── __init__.py
├── Tools/
│   ├── rmsnorm/
│   │   ├── rmsnorm.cu         Fused Add-RMSNorm CUDA kernel
│   │   └── rmsnorm.cuh
│   └── swiglu/
│       ├── swiglu.cu          Fused SwiGLU CUDA kernel
│       ├── swiglu.cuh
│       ├── swiglu_binding.cpp
│       ├── FusedSwiglu.py
│       └── __init__.py
├── Inference/
│   └── Inference.py           Autoregressive inference script
├── weights/
│   └── download_weights.py    Downloads instruct weights from HuggingFace
└── setup.py                   Builds the CUDA extensions
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision safetensors transformers huggingface_hub pillow numpy
```

### 2. Build the CUDA extensions

```bash
python setup.py build_ext --inplace
```

The build compiles two extensions: `rmsnorm` and `swiglu_fused`. Both are optional — the model detects their presence at runtime and falls back to PyTorch implementations if they are not available.

### 3. Download weights

LLaMA models are gated on HuggingFace. Authenticate once, then run the download script:

```bash
huggingface-cli login
python weights/download_weights.py
```

Weights are saved to `weights/Llama-3.2-11B-Vision-Instruct/` by default. To download a different variant:

```bash
python weights/download_weights.py \
    --model-id  meta-llama/Llama-3.2-90B-Vision-Instruct \
    --output-dir weights/Llama-3.2-90B-Vision-Instruct
```

---

## Inference

```bash
python Inference/Inference.py \
    --image   path/to/image.jpg \
    --prompt  "Describe what is in this image." \
    --hf-weights weights/Llama-3.2-11B-Vision-Instruct
```

### Generation options

| Flag | Default | Description |
|---|---|---|
| `--max-new-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.0 | Sampling temperature. 0 = greedy |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--top-k` | 50 | Top-k sampling |
| `--dtype` | auto | `float16`, `bfloat16`, or `float32` |
| `--cpu` | — | Force CPU inference |

---

## Programmatic usage

### Build a model from config

```python
from Model import MLLAMAConfig, MllamaForConditionalGeneration

vision_cfg = {
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "image_size": 560,
    "patch_size": 14,
}
text_cfg = {
    "vocab_size": 128256,
    "hidden_size": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14336,
    "n_kv_groups": 8,
}

cfg   = MLLAMAConfig(vision_config=vision_cfg, text_config=text_cfg, projection_dim=4096)
model = MllamaForConditionalGeneration(cfg)
```

### Load pretrained weights

```python
from Model.utils import load_hf_model

model, tokenizer = load_hf_model(
    "weights/Llama-3.2-11B-Vision-Instruct",
    device="cuda",
)
model.eval()
```

### Forward pass

```python
outputs = model(
    input_ids=input_ids,         # [B, seq_len]
    pixel_values=pixel_values,   # [B, 3, H, W]
    attention_mask=attention_mask,
)

logits = outputs["logits"]       # [B, seq_len, vocab_size]
loss   = outputs["loss"]         # set when labels are provided
```

---

## CUDA kernels

| Kernel | Source | Used by |
|---|---|---|
| Fused Add-RMSNorm (forward + backward) | `Tools/rmsnorm/rmsnorm.cu` | `LLAMARMSNorm` |
| Fused SwiGLU (forward + backward) | `Tools/swiglu/swiglu.cu` | `FusedFeedforward` |

Both kernels support `float16` and `bfloat16`, with the SwiGLU kernel using tiled shared memory for the optimised forward path. They activate automatically when the input is on a CUDA device with a half-precision dtype; otherwise the model uses the PyTorch fallback transparently.

---

## LoRA fine-tuning

`Linear_LORA` in `model.py` wraps any `nn.Linear` with frozen base weights and trainable low-rank adapters. To convert the language model for parameter-efficient fine-tuning:

```python
from Model.model import Linear_LORA

def apply_lora(module, rank=16, alpha=32, dropout=0.05):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            lora = Linear_LORA(child.in_features, child.out_features, rank, alpha, dropout)
            lora.linear.weight.data.copy_(child.weight.data)
            setattr(module, name, lora)
        else:
            apply_lora(child, rank, alpha, dropout)

apply_lora(model.language_model)
```

Only the `lora_a` and `lora_b` adapter weights require gradients. Save just those for a compact checkpoint.

---

## License

MIT