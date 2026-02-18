from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from .model import MLLAMAConfig, MllamaForConditionalGeneration


HF_TO_LOCAL_KEY_SUBSTRINGS = OrderedDict(
    {
        "multi_modal_projector.linear_1": "multi_modal_projector.linear",
        "language_model.model.embed_tokens": "language_model.model.tok_emb",
        "language_model.lm_head": "language_model.lm_head",
        "language_model.model.layers": "language_model.model.trf_blocks",
        "self_attn.q_proj": "att.W_query",
        "self_attn.k_proj": "att.W_key",
        "self_attn.v_proj": "att.W_value",
        "self_attn.o_proj": "att.out_proj",
        "input_layernorm": "norm1",
        "post_attention_layernorm": "norm2",
        "mlp.gate_proj": "ff.swiglu.w_gate",
        "mlp.up_proj": "ff.swiglu.w_up",
        "mlp.down_proj": "ff.w_down",
        "language_model.model.norm": "language_model.model.final_norm",
        "vision_model": "vision_model.vision_model",
        "embeddings.patch_embedding": "embedding.patch_embedding",
        "embeddings.position_embedding": "embedding.position_embedding",
        "encoder.layers": "encoder.layers",
        "self_attn": "self_attn",
        "layer_norm1": "layernorm1",
        "layer_norm2": "layernorm2",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
        "post_layernorm": "layernorm",
    }
)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_safetensors(model_path: str) -> Dict[str, torch.Tensor]:
    tensor_files = sorted(
        [
            os.path.join(model_path, file_name)
            for file_name in os.listdir(model_path)
            if file_name.endswith(".safetensors")
        ]
    )

    if not tensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found under '{model_path}'. "
            "Download the model checkpoint first."
        )

    tensors: Dict[str, torch.Tensor] = {}
    for safetensor_file in tensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as file_handle:
            for key in file_handle.keys():
                tensors[key] = file_handle.get_tensor(key)
    return tensors


def _hf_text_to_local_config(text_cfg: Dict, pad_token_id: int | None) -> Dict:
    return {
        "vocab_size": text_cfg["vocab_size"],
        "hidden_size": text_cfg["hidden_size"],
        "context_length": text_cfg.get("max_position_embeddings", 131072),
        "n_heads": text_cfg["num_attention_heads"],
        "n_layers": text_cfg["num_hidden_layers"],
        "hidden_dim": text_cfg["intermediate_size"],
        "max_position_embeddings": text_cfg.get("max_position_embeddings", 2048),
        "n_kv_groups": text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]),
        "rope_base": text_cfg.get("rope_theta", 500000.0),
        "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-5),
        "pad_token_index": pad_token_id,
    }


def _hf_vision_to_local_config(vision_cfg: Dict) -> Dict:
    return {
        "hidden_size": vision_cfg["hidden_size"],
        "intermediate_size": vision_cfg["intermediate_size"],
        "num_hidden_layers": vision_cfg["num_hidden_layers"],
        "num_attention_heads": vision_cfg["num_attention_heads"],
        "num_channels": vision_cfg.get("num_channels", 3),
        "image_size": vision_cfg["image_size"],
        "patch_size": vision_cfg["patch_size"],
        "layer_norm_eps": vision_cfg.get("layer_norm_eps", 1e-6),
        "attention_dropout": vision_cfg.get("attention_dropout", 0.0),
    }


def _build_local_config(config_dict: Dict, pad_token_id: int | None) -> MLLAMAConfig:
    text_cfg = _hf_text_to_local_config(config_dict["text_config"], pad_token_id)
    vision_cfg = _hf_vision_to_local_config(config_dict["vision_config"])

    return MLLAMAConfig(
        ignore_index=config_dict.get("ignore_index", -100),
        image_token_index=config_dict["image_token_index"],
        vocab_size=config_dict.get("vocab_size", text_cfg["vocab_size"]),
        projection_dim=config_dict.get("vision_config", {}).get(
            "projection_dim", text_cfg["hidden_size"]
        ),
        hidden_size=text_cfg["hidden_size"],
        vision_config=vision_cfg,
        text_config=text_cfg,
        pad_token_index=pad_token_id,
    )


def _translate_weight_key(hf_key: str) -> str | None:
    # Keys belonging to heads not implemented in this lightweight inference stack.
    unsupported_prefixes = (
        "vision_model.global_transformer",
        "vision_model.vision_model.tile_",
        "vision_model.vision_model.pre_",
        "vision_model.vision_model.gated_",
        "language_model.model.rotary_emb",
    )
    if hf_key.startswith(unsupported_prefixes) or ".cross_attn" in hf_key:
        return None

    translated = hf_key
    for src, dst in HF_TO_LOCAL_KEY_SUBSTRINGS.items():
        translated = translated.replace(src, dst)

    # Translate fused SwiGLU parameters into our custom parameter names.
    translated = translated.replace("ff.swiglu.w_gate.weight", "ff.swiglu.w_gate")
    translated = translated.replace("ff.swiglu.w_up.weight", "ff.swiglu.w_up")

    # Drop bias terms for modules created with bias=False in the local implementation.
    if translated.endswith(".bias"):
        return None

    return translated


def _convert_hf_state_dict_to_local(
    hf_tensors: Dict[str, torch.Tensor],
    model: MllamaForConditionalGeneration,
) -> Tuple[Dict[str, torch.Tensor], list[str], list[str]]:
    target_state = model.state_dict()

    converted_state: Dict[str, torch.Tensor] = {}
    skipped_unmapped: list[str] = []

    for hf_key, tensor in hf_tensors.items():
        local_key = _translate_weight_key(hf_key)
        if local_key is None:
            skipped_unmapped.append(hf_key)
            continue

        if local_key not in target_state:
            skipped_unmapped.append(hf_key)
            continue

        if target_state[local_key].shape != tensor.shape:
            skipped_unmapped.append(
                f"{hf_key} (shape {tuple(tensor.shape)} -> expected {tuple(target_state[local_key].shape)})"
            )
            continue

        converted_state[local_key] = tensor

    missing_after_conversion = [
        key for key in target_state.keys() if key not in converted_state
    ]

    return converted_state, skipped_unmapped, missing_after_conversion


def load_hf_model(model_path: str, device: str) -> Tuple[MllamaForConditionalGeneration, AutoTokenizer]:
    """
    Load Hugging Face Llama 3.2 multimodal weights into the local custom architecture.

    This utility performs explicit key translation from Hugging Face naming conventions
    into this repository's lightweight module names and tensor layouts.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    config_path = os.path.join(model_path, "config.json")
    config_dict = _read_json(config_path)
    model_config = _build_local_config(config_dict, tokenizer.pad_token_id)

    model = MllamaForConditionalGeneration(model_config).to(device)

    hf_tensors = _read_safetensors(model_path)
    converted_state, skipped, missing = _convert_hf_state_dict_to_local(hf_tensors, model)

    # Load what we can; strict=False allows unsupported branches to stay with init weights.
    load_result = model.load_state_dict(converted_state, strict=False)
    model.tie_weights()

    # Surface useful diagnostics for debugging incomplete mappings.
    if skipped:
        print(f"[load_hf_model] Skipped {len(skipped)} source tensors that do not map to local modules.")
    unresolved = sorted(set(missing + list(load_result.missing_keys)))
    if unresolved:
        print(f"[load_hf_model] Missing {len(unresolved)} target tensors after conversion.")

    return model, tokenizer
