import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with Llama 3.2 Vision weights from Hugging Face.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Prompt/question for the model.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
        help="Hugging Face model id (must be a Llama 3.2 Vision checkpoint).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value when temperature > 0.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype to request from Transformers.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_messages(prompt: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def run_inference(args: argparse.Namespace) -> str:
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch_dtype = resolve_dtype(args.dtype)

    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    image = load_image(args.image)
    messages = build_messages(args.prompt)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    model_inputs = processor(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    output = model.generate(**model_inputs, **generate_kwargs)
    continuation = output[:, model_inputs["input_ids"].shape[-1] :]
    decoded = processor.decode(continuation[0], skip_special_tokens=True)
    return decoded.strip()


def main() -> None:
    args = parse_args()
    result = run_inference(args)
    print(result)


if __name__ == "__main__":
    main()
