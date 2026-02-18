<<<<<<< HEAD
from PIL import Image
import torch
import fire

from ..Model.processing_mllama import MllamaImageProcessor
from ..Model.model import KVCache,MllamaForConditionalGeneration

=======
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from Model.model import MLLAMAConfig, MllamaForConditionalGeneration
from Model.processing_mllama import MllamaImageProcessor


def build_default_config():
    vision_config = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "image_size": 224,
        "patch_size": 16,
    }

    text_config = {
        "vocab_size": 128257,
        "hidden_size": 1024,
        "n_heads": 16,
        "n_layers": 8,
        "hidden_dim": 4096,
        "max_position_embeddings": 2048,
        "dtype": torch.float32,
    }

    return MLLAMAConfig(
        vision_config=vision_config,
        text_config=text_config,
        projection_dim=text_config["hidden_size"],
        image_token_index=128256,
    )


def load_checkpoint_if_available(model, checkpoint_path):
    if checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    config = build_default_config()
    model = MllamaForConditionalGeneration(config).to(device)
    model.eval()

    load_checkpoint_if_available(model, args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    num_image_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    processor = MllamaImageProcessor(
        tokenizer=tokenizer,
        num_image_token=num_image_tokens,
        image_size=config.vision_config.image_size,
    )

    image = Image.open(args.image).convert("RGB")
    batch = processor([args.prompt], [image], padding=True)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_value"].to(device=device, dtype=model.language_model.model.tok_emb.weight.dtype)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

    next_token = outputs["logits"][0, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids[0], next_token.cpu()], dim=0)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    print(decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--tokenizer", default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--checkpoint", default=None, help="Optional .pt checkpoint path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    run(parser.parse_args())
>>>>>>> d672be6e3d4a6d1ba57865815bda29658ae896ac
