# LLaMA 3.2 Multimodal

This repository contains a **custom PyTorch implementation** of a LLaMA 3.2-style multimodal stack (vision encoder + language model), plus a simple inference CLI.

The project now supports two practical ways to run with Hugging Face checkpoints:

1. **Direct Transformers runtime** (official `transformers` classes).
2. **Custom local runtime + explicit Hugging Face weight import** via `Model/utils.py`.

---

## What was updated

After reviewing the codebase end-to-end, the local loader was updated so it can:

- read Hugging Face `.safetensors` shards,
- convert Hugging Face config fields into local model config classes,
- map Hugging Face parameter names to this repo’s internal module names,
- skip unsupported/unimplemented branches safely,
- tie LM head weights to token embeddings to match common causal-LM behavior.

This makes it possible to load a compatible Hugging Face multimodal checkpoint into the local model and assign available weights deterministically.

---

## Project layout

- `Inference/Inference.py` — CLI using Hugging Face `MllamaForConditionalGeneration` + `AutoProcessor`.
- `Model/model.py` — local multimodal architecture (vision + text).
- `Model/utils.py` — Hugging Face -> local model loading and key translation.
- `Model/processing_mllama.py` — custom image/prompt preprocessing utilities.
- `Tools/` — optional CUDA fused kernels (RMSNorm, SwiGLU).

---

## Requirements

- Python 3.10+
- PyTorch
- `transformers`
- `safetensors`
- `Pillow`
- `accelerate` (recommended for larger checkpoints)

Install:

```bash
pip install --upgrade torch transformers safetensors pillow accelerate
```

If loading gated Meta checkpoints, login first:

```bash
huggingface-cli login
```

---

## Option A: Run inference with official Transformers model (recommended baseline)

Use the included CLI:

```bash
python Inference/Inference.py \
  --image /path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --model-id meta-llama/Llama-3.2-11B-Vision-Instruct
```

Useful flags:

- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--cpu`
- `--dtype auto|float16|bfloat16|float32`

---

## Option B: Load Hugging Face weights into the local custom model

`Model/utils.py` exposes:

```python
load_hf_model(model_path: str, device: str)
```

### What this loader does

1. Loads tokenizer from `model_path`.
2. Reads `config.json` and converts HF schema -> local `MLLAMAConfig` / `LLAMA32Config` / `SiglipVisionConfig`.
3. Reads all `.safetensors` tensors.
4. Translates keys (HF naming -> local naming).
5. Loads compatible tensors with `strict=False`.
6. Ties LM head and token embedding weights.
7. Prints diagnostics about skipped/missing tensors.

### Important compatibility notes

Because this repository’s architecture is a simplified custom implementation, some Hugging Face tensors may be skipped if:

- that branch is not implemented in the local code,
- key names do not have a matching local module,
- parameter shapes differ.

This is expected for non-overlapping submodules and allows incremental support while still importing the majority of compatible weights.

---

## Minimal local loader example

```python
from Model.utils import load_hf_model

model, tokenizer = load_hf_model(
    model_path="/path/to/downloaded/hf/model",
    device="cuda",
)
```

---

## Deep-code review highlights

- **Text stack**: grouped-query attention, rotary embeddings, RMSNorm, fused SwiGLU feed-forward path.
- **Vision stack**: SigLIP-style patch embedding + transformer encoder blocks.
- **Multimodal merge**: image patch embeddings replace `<image>` token slots in text embeddings.
- **Inference path**: maintained both official HF direct path and local-path HF weight assignment tooling.

---

## Troubleshooting

- **`ModuleNotFoundError: transformers`**: install `transformers` in your environment.
- **Missing model access**: ensure your HF token can access the checkpoint.
- **Large GPU memory usage**: start with `--dtype bfloat16` (or `float16`) and lower generation length.
- **Loader reports skipped tensors**: expected when local architecture does not yet include every HF branch.

---

## License

See `LICENSE`.
