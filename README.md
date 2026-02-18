# LLaMA 3.2 Multimodal (Hugging Face Inference)

This repository is now focused on **running inference with official Llama 3.2 Vision checkpoints from Hugging Face**.

Instead of maintaining a separate custom architecture, the inference path uses `transformers`' built-in `MllamaForConditionalGeneration` + `AutoProcessor`, which lets you directly load Meta's published weights and run image+text prompts.

## What this repo now provides

- A simple CLI at `Inference/Inference.py` to:
  - load a Llama 3.2 Vision model from Hugging Face,
  - read an input image,
  - build a chat-style multimodal prompt,
  - generate and print an answer.

## Requirements

- Python 3.10+
- A Hugging Face account with access to the model you choose (for Meta checkpoints)
- `transformers`, `torch`, `Pillow`, `accelerate`

Install:

```bash
pip install --upgrade torch transformers accelerate pillow
```

If needed, log in to Hugging Face:

```bash
huggingface-cli login
```

## Supported model IDs

Default model used by the script:

- `meta-llama/Llama-3.2-11B-Vision-Instruct`

You can override with `--model-id` (for example, a different Llama 3.2 Vision variant you have access to).

## Quick start

```bash
python Inference/Inference.py \
  --image /path/to/image.jpg \
  --prompt "Describe this image in detail."
```

## CLI options

```bash
python Inference/Inference.py --help
```

Main flags:

- `--image` (required): input image path
- `--prompt` (required): user prompt/question
- `--model-id`: Hugging Face model id (default: `meta-llama/Llama-3.2-11B-Vision-Instruct`)
- `--max-new-tokens`: generation length (default: `128`)
- `--temperature`: `0.0` for greedy; `>0` enables sampling
- `--top-p`: top-p value used when sampling
- `--cpu`: force CPU inference
- `--dtype`: `auto | float16 | bfloat16 | float32`

## Example with sampling

```bash
python Inference/Inference.py \
  --image ./example.jpg \
  --prompt "What is happening in this scene?" \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9
```

## Notes

- First run will download model + processor weights from Hugging Face.
- Vision checkpoints are large; GPU inference is strongly recommended.
- If you see access errors, verify your Hugging Face token and model permissions.

## Project structure

- `Inference/Inference.py`: Hugging Face loading + multimodal generation entrypoint.
- `Model/` and `Tools/`: previous custom implementation artifacts kept in-repo, but the supported path for running Llama 3.2 Vision weights is now the Hugging Face inference CLI above.
