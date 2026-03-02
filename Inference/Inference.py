import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for LLaMA-3.2 Vision VLM."
    )
    parser.add_argument("--image",  required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Text prompt or question.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model repo ID (used when --hf-weights is not set).",
    )
    parser.add_argument(
        "--hf-weights",
        default=None,
        help=(
            "Path to a local HF checkpoint directory downloaded via "
            "weights/download_weights.py. When set, uses the custom VLM architecture."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 = greedy decoding (default).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold (only used when temperature > 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (only used when temperature > 0).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when a GPU is available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype for HF transformers mode (default: auto).",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }.get(name, "auto")


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def select_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens once cumulative probability exceeds top_p
        sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def run_custom_inference(args: argparse.Namespace) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from Model.utils import load_hf_model
    from Model.processing_mllama import MllamaImageProcessor
    from Model.model import KVCache

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    print(f"Loading model from: {args.hf_weights}")
    model, tokenizer = load_hf_model(args.hf_weights, device=device)
    model.eval()

    num_image_tokens = model.config.text_config.num_image_tokens
    image_size       = model.config.vision_config.image_size
    processor        = MllamaImageProcessor(tokenizer, num_image_tokens, image_size)

    image  = load_image(args.image)
    inputs = processor([args.prompt], [image], padding=True)

    pixel_values   = inputs["pixel_values"].to(device)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    eos_token_id = tokenizer.eos_token_id
    generated    = []
    kv_cache     = KVCache()

    with torch.no_grad():
        # Prefill — process the full prompt + image together
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        # Sample the first new token from the last position in the prefill
        next_token = select_next_token(
            outputs["logits"][0, -1],
            args.temperature, args.top_p, args.top_k,
        )
        generated.append(next_token.item())

        # Decode loop — one token at a time, reusing the KV cache
        for _ in range(args.max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            # Extend the attention mask by one position for the new token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

            outputs = model(
                input_ids=next_token.unsqueeze(0),   # [1, 1]
                pixel_values=None,                    
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

            next_token = select_next_token(
                outputs["logits"][0, -1],
                args.temperature, args.top_p, args.top_k,
            )
            generated.append(next_token.item())

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_hf_inference(args: argparse.Namespace) -> str:
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    device      = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch_dtype = resolve_dtype(args.dtype)

    print(f"Loading HF model: {args.model_id}")
    model     = MllamaForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    image    = load_image(args.image)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
        }
    ]
    prompt       = processor.apply_chat_template(messages, add_generation_prompt=True)
    model_inputs = processor(
        image, prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"]       = args.top_p
        gen_kwargs["top_k"]       = args.top_k

    output       = model.generate(**model_inputs, **gen_kwargs)
    continuation = output[:, model_inputs["input_ids"].shape[-1]:]
    return processor.decode(continuation[0], skip_special_tokens=True).strip()


def main() -> None:
    args   = parse_args()
    result = run_custom_inference(args) if args.hf_weights else run_hf_inference(args)
    print(result)


if __name__ == "__main__":
    main()