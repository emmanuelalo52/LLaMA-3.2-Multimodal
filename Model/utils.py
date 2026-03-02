"""
utils.py — Weight-loading utilities for the plain-ViT VLM.
Updated key translation for VisionEncoder (no SigLIP).
"""

from __future__ import annotations
import json, os
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from .model import MLLAMAConfig, MllamaForConditionalGeneration


HF_TO_LOCAL_KEY_SUBSTRINGS: OrderedDict[str, str] = OrderedDict({
    # Projector
    "multi_modal_projector.linear_1":       "multi_modal_projector.linear",
    # LM
    "language_model.model.embed_tokens":    "language_model.model.tok_emb",
    "language_model.lm_head":               "language_model.lm_head",
    "language_model.model.layers":          "language_model.model.trf_blocks",
    "self_attn.q_proj":                     "att.W_query",
    "self_attn.k_proj":                     "att.W_key",
    "self_attn.v_proj":                     "att.W_value",
    "self_attn.o_proj":                     "att.out_proj",
    "input_layernorm":                      "norm1",
    "post_attention_layernorm":             "norm2",
    "mlp.gate_proj":                        "ff.swiglu.w_gate",
    "mlp.up_proj":                          "ff.swiglu.w_up",
    "mlp.down_proj":                        "ff.w_down",
    "language_model.model.norm":            "language_model.model.final_norm",
    # Vision encoder (plain ViT)
    "vision_model.vision_model.patch_embedding":    "vision_model.embeddings.patch_embedding",
    "vision_model.vision_model.position_embedding": "vision_model.embeddings.position_embedding",
    "vision_model.vision_model.encoder.layers":     "vision_model.encoder.layers",
    "layer_norm1":                          "layernorm1",
    "layer_norm2":                          "layernorm2",
    "self_attn.out_proj":                   "self_attn.out_proj",
    "mlp.fc1":                              "mlp.fc1",
    "mlp.fc2":                              "mlp.fc2",
    "vision_model.vision_model.post_layernorm": "vision_model.post_layernorm",
})


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_safetensors(model_path: str) -> Dict[str, torch.Tensor]:
    tensor_files = sorted(
        os.path.join(model_path, fn)
        for fn in os.listdir(model_path)
        if fn.endswith(".safetensors")
    )
    if not tensor_files:
        raise FileNotFoundError(
            f"No .safetensors files under '{model_path}'. "
            "Run weights/download_weights.py first."
        )
    tensors: Dict[str, torch.Tensor] = {}
    for sf in tensor_files:
        with safe_open(sf, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensors[key] = fh.get_tensor(key)
    return tensors


def _hf_text_to_local_config(tc: Dict, pad_token_id) -> Dict:
    return {
        "vocab_size": tc["vocab_size"], "hidden_size": tc["hidden_size"],
        "context_length": tc.get("max_position_embeddings", 131072),
        "n_heads": tc["num_attention_heads"], "n_layers": tc["num_hidden_layers"],
        "hidden_dim": tc["intermediate_size"],
        "max_position_embeddings": tc.get("max_position_embeddings", 2048),
        "n_kv_groups": tc.get("num_key_value_heads", tc["num_attention_heads"]),
        "rope_base": tc.get("rope_theta", 500000.0),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-5),
        "pad_token_index": pad_token_id,
    }


def _hf_vision_to_local_config(vc: Dict) -> Dict:
    return {
        "hidden_size": vc["hidden_size"], "intermediate_size": vc["intermediate_size"],
        "num_hidden_layers": vc["num_hidden_layers"],
        "num_attention_heads": vc["num_attention_heads"],
        "num_channels": vc.get("num_channels", 3),
        "image_size": vc["image_size"], "patch_size": vc["patch_size"],
        "layer_norm_eps": vc.get("layer_norm_eps", 1e-6),
        "attention_dropout": vc.get("attention_dropout", 0.0),
    }


def _build_local_config(cfg: Dict, pad_token_id) -> MLLAMAConfig:
    tc = _hf_text_to_local_config(cfg["text_config"], pad_token_id)
    vc = _hf_vision_to_local_config(cfg["vision_config"])
    return MLLAMAConfig(
        ignore_index=cfg.get("ignore_index", -100),
        image_token_index=cfg["image_token_index"],
        vocab_size=cfg.get("vocab_size", tc["vocab_size"]),
        projection_dim=cfg.get("vision_config", {}).get("projection_dim", tc["hidden_size"]),
        hidden_size=tc["hidden_size"],
        vision_config=vc, text_config=tc, pad_token_index=pad_token_id,
    )


def _translate_weight_key(hf_key: str) -> str | None:
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
    translated = translated.replace("ff.swiglu.w_gate.weight", "ff.swiglu.w_gate")
    translated = translated.replace("ff.swiglu.w_up.weight",   "ff.swiglu.w_up")
    if translated.endswith(".bias"):
        return None
    return translated


def _convert_hf_state_dict_to_local(
    hf_tensors: Dict[str, torch.Tensor],
    model: MllamaForConditionalGeneration,
) -> Tuple[Dict[str, torch.Tensor], list, list]:
    target_state = model.state_dict()
    converted_state: Dict[str, torch.Tensor] = {}
    skipped: list = []
    for hf_key, tensor in hf_tensors.items():
        local_key = _translate_weight_key(hf_key)
        if local_key is None or local_key not in target_state:
            skipped.append(hf_key); continue
        if target_state[local_key].shape != tensor.shape:
            skipped.append(f"{hf_key} (shape mismatch)"); continue
        converted_state[local_key] = tensor
    missing = [k for k in target_state if k not in converted_state]
    return converted_state, skipped, missing


def load_hf_model(
    model_path: str, device: str
) -> Tuple[MllamaForConditionalGeneration, AutoTokenizer]:
    """Load HF Llama-3.2 Vision instruct weights into the local VLM architecture."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    cfg_dict = _read_json(os.path.join(model_path, "config.json"))
    model_cfg = _build_local_config(cfg_dict, tokenizer.pad_token_id)
    model = MllamaForConditionalGeneration(model_cfg).to(device)
    hf_tensors = _read_safetensors(model_path)
    converted, skipped, missing = _convert_hf_state_dict_to_local(hf_tensors, model)
    load_result = model.load_state_dict(converted, strict=False)
    model.tie_weights()
    if skipped:
        print(f"[load_hf_model] Skipped {len(skipped)} source keys.")
    unresolved = sorted(set(missing + list(load_result.missing_keys)))
    if unresolved:
        print(f"[load_hf_model] {len(unresolved)} target keys missing after conversion.")
    return model, tokenizer