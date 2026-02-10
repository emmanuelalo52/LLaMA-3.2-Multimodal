# LLaMA 3.2 Multimodal Implementation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

*A modular PyTorch implementation of a multimodal vision-language model inspired by Meta's LLaMA 3.2 architecture*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📋 Overview

This repository provides a clean, research-friendly PyTorch implementation of a multimodal large language model that combines:

- **Vision Encoder**: SigLIP-based Vision Transformer for processing images into patch embeddings
- **Language Model**: LLaMA-style decoder with Group-Query Attention (GQA), Rotary Position Embeddings (RoPE), and RMSNorm
- **Multimodal Fusion**: Cross-attention mechanism to integrate visual and textual information
- **LoRA Support**: Low-Rank Adaptation for efficient fine-tuning with minimal trainable parameters

### Why This Implementation?

- **🎯 Educational**: Clear, well-documented code designed for learning and experimentation
- **🔧 Modular**: Easy to swap components (vision encoder, language model, fusion strategy)
- **⚡ Efficient**: Supports KV-caching, mixed-precision training, and LoRA fine-tuning
- **🚀 Production-Ready**: Includes utilities for preprocessing, tokenization, and checkpoint management

---

## ✨ Features

### Model Architecture
- **SigLIP Vision Transformer**: Efficient patch-based image encoding with learnable position embeddings
- **LLaMA-3 Language Model**: State-of-the-art transformer decoder with:
  - Group-Query Attention (GQA) for improved inference efficiency
  - Rotary Position Embeddings (RoPE) for better length extrapolation
  - RMSNorm for stable training
  - SwiGLU activation functions
- **Multimodal Projection**: Learnable adapter to align vision and language representations

### Training & Fine-Tuning
- **LoRA (Low-Rank Adaptation)**: Memory-efficient fine-tuning
- **Mixed Precision**: FP16/BF16 training support
- **KV-Cache**: Efficient autoregressive generation
- **Gradient Checkpointing**: Reduced memory footprint for large models

### Utilities
- **Image Preprocessing**: Built-in normalization and resizing for SigLIP
- **Tokenization**: Support for HuggingFace tokenizers with special image tokens
- **Checkpoint Management**: SafeTensors format for secure model serialization

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/emmanuelalo52/LLaMA-3.2-Multimodal.git
cd LLaMA-3.2-Multimodal

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers safetensors tiktoken
pip install numpy pillow
```

### Development Installation

```bash
pip install -e .  # Editable install for development
```

---

## 🚀 Quick Start

### Basic Inference

```python
import torch
from PIL import Image
from transformers import LlamaTokenizer

from Model.model import MllamaForConditionalGeneration, MLLAMAConfig
from Model.processing_mllama import MllamaImageProcessor

# 1. Initialize model configuration
vision_config = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
}

text_config = {
    "vocab_size": 128256,
    "hidden_size": 4096,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "max_position_embeddings": 2048,
}

config = MLLAMAConfig(
    vision_config=vision_config,
    text_config=text_config,
    projection_dim=2048
)

# 2. Create model
model = MllamaForConditionalGeneration(config)
model.eval()

# 3. Load tokenizer and image processor
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
num_image_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
image_processor = MllamaImageProcessor(
    tokenizer=tokenizer,
    num_image_token=num_image_tokens,
    image_size=config.vision_config.image_size
)

# 4. Prepare input
image = Image.open("path/to/image.jpg").convert("RGB")
prompt = "Describe this image in detail."

batch = image_processor([prompt], [image], padding=True)
pixel_values = batch["pixel Value"]  # Note: space in key name
input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]

# 5. Generate response
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )

logits = outputs["logits"]
predicted_ids = logits.argmax(dim=-1)
response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
print(f"Model: {response}")
```

### Loading Pretrained Weights

```python
from safetensors.torch import load_file

# Load pretrained weights
state_dict = load_file("path/to/checkpoint.safetensors")
model.load_state_dict(state_dict, strict=False)
```

---

## 📚 Documentation

### Project Structure

```
LLaMA-3.2-Multimodal/
├── Model/
│   ├── model.py                 # Core multimodal model implementation
│   ├── siglip.py                # SigLIP vision transformer
│   └── processing_mllama.py     # Image preprocessing utilities
├── Tools/
│   └── ...                      # CUDA kernels and optimization utilities
├── build/
│   └── ...                      # Compiled extensions
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

### Key Components

#### 1. Model Architecture (`Model/model.py`)

**MLLAMAConfig**: Configuration class for the entire multimodal model
```python
config = MLLAMAConfig(
    vision_config=vision_config,   # SigLIP configuration
    text_config=text_config,       # LLaMA configuration
    projection_dim=2048            # Vision-to-language projection dimension
)
```

**MllamaForConditionalGeneration**: Main model class
- Combines vision encoder, language decoder, and multimodal projection
- Handles image-text fusion via cross-attention
- Supports loss computation with flexible masking strategies

**Key Features**:
- `Linear_LORA`: LoRA wrapper for efficient fine-tuning
- `GroupQueryAttention`: GQA implementation with KV-cache support
- `LLAMARotaryEmbedding`: RoPE for position encoding
- `MultiModalProjector`: Vision-language alignment layer

#### 2. Vision Encoder (`Model/siglip.py`)

**SigLIP Architecture**:
- Patch embedding via Conv2D (14x14 or 16x16 patches)
- Learnable position embeddings
- Multi-head self-attention transformer encoder
- Output: `[batch, num_patches, hidden_dim]`

**SiglipVisionConfig**:
```python
vision_config = {
    "hidden_size": 1152,           # Model dimension
    "intermediate_size": 4304,     # FFN dimension
    "num_hidden_layers": 27,       # Transformer layers
    "num_attention_heads": 16,     # Attention heads
    "image_size": 384,             # Input image size
    "patch_size": 14,              # Patch size
}
```

#### 3. Image Processing (`Model/processing_mllama.py`)

**MllamaImageProcessor**:
- Adds special `<image>` token to vocabulary
- Resizes and normalizes images (ImageNet statistics)
- Creates image placeholder tokens in text prompts
- Returns dict with `"pixel Value"` (note the space), `input_ids`, `attention_mask`

**Usage**:
```python
processor = MllamaImageProcessor(
    tokenizer=tokenizer,
    num_image_token=196,  # (14x14 patches)
    image_size=224
)

batch = processor(
    text=["Describe this image"],
    images=[PIL_image],
    padding=True
)
```

---

## 🎓 Advanced Usage

### LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding trainable low-rank matrices to frozen model weights.

```python
from Model.model import Linear_LORA

def convert_to_lora(model, rank=8, alpha=16, dropout=0.1):
    """Convert linear layers to LoRA for efficient fine-tuning"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Target specific layers (e.g., attention projections)
            if any(target in name for target in ['q_proj', 'v_proj', 'o_proj']):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                
                # Create LoRA wrapper
                lora_layer = Linear_LORA(
                    in_dim=module.in_features,
                    out_dim=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Copy original weights
                lora_layer.linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora_layer.linear.bias.data.copy_(module.bias.data)
                
                # Replace module
                setattr(parent, child_name, lora_layer)
    
    # Freeze base model
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    return model

# Apply LoRA
model = convert_to_lora(model, rank=8, alpha=16)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

### Training Loop Example

```python
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# Setup
model = model.cuda()
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.01
)
scaler = GradScaler()  # For mixed precision

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        pixel_values = batch["pixel_values"].cuda()
        labels = batch["labels"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Custom Dataset Integration

```python
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, data_path, image_processor, tokenizer):
        self.data = self.load_data(data_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        text = item['text']
        
        # Process inputs
        batch = self.image_processor([text], [image], padding=True)
        
        # Create labels (for training)
        labels = batch['input_ids'].clone()
        # Mask image tokens in loss computation
        labels[labels == self.image_processor.image_token_id] = -100
        
        return {
            'input_ids': batch['input_ids'].squeeze(0),
            'pixel_values': batch['pixel Value'].squeeze(0),
            'attention_mask': batch['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def load_data(self, path):
        # Load your dataset here
        pass

# Create dataloader
dataset = MultimodalDataset('data.json', image_processor, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

---

## 💡 Examples

### Example 1: Visual Question Answering

```python
from PIL import Image
import torch

# Load image
image = Image.open("examples/cityscape.jpg")
question = "How many cars are visible in this image?"

# Process
batch = image_processor([question], [image])
outputs = model.generate(
    input_ids=batch["input_ids"],
    pixel_values=batch["pixel Value"],
    max_new_tokens=50,
    temperature=0.7
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Q: {question}")
print(f"A: {answer}")
```

### Example 2: Image Captioning

```python
# Generate detailed caption
image = Image.open("examples/landscape.jpg")
prompt = "Provide a detailed description of this image, including colors, objects, and atmosphere."

batch = image_processor([prompt], [image])
outputs = model.generate(
    input_ids=batch["input_ids"],
    pixel_values=batch["pixel Value"],
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9
)

caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

### Example 3: Batch Processing

```python
# Process multiple images
images = [Image.open(f"images/{i}.jpg") for i in range(4)]
prompts = ["Describe this image."] * 4

batch = image_processor(prompts, images, padding=True)
outputs = model(**batch)

for i, logits in enumerate(outputs["logits"]):
    pred = logits.argmax(dim=-1)
    text = tokenizer.decode(pred, skip_special_tokens=True)
    print(f"Image {i}: {text}")
```

---

## ⚙️ Configuration

### Model Configuration Options

```python
# Vision Encoder Configuration
vision_config = {
    "hidden_size": 1152,           # Model dimension
    "intermediate_size": 4304,     # FFN dimension  
    "num_hidden_layers": 27,       # Number of transformer layers
    "num_attention_heads": 16,     # Number of attention heads
    "num_channels": 3,             # RGB channels
    "image_size": 384,             # Input image resolution
    "patch_size": 14,              # Patch size (14x14)
}

# Language Model Configuration
text_config = {
    "vocab_size": 128256,          # Vocabulary size
    "hidden_size": 4096,           # Model dimension
    "n_heads": 32,                 # Number of attention heads
    "n_kv_heads": 8,               # Number of KV heads (GQA)
    "n_layers": 32,                # Number of transformer layers
    "hidden_dim": 14336,           # FFN dimension
    "max_position_embeddings": 131072,  # Max sequence length
    "rope_theta": 500000.0,        # RoPE base frequency
    "rms_norm_eps": 1e-5,          # RMSNorm epsilon
}

# Full Model Configuration
config = MLLAMAConfig(
    vision_config=vision_config,
    text_config=text_config,
    projection_dim=2048,           # Vision-to-language projection
    image2text_projection_bias=True
)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `KeyError: 'pixel Value'`
- **Solution**: The image processor returns `"pixel Value"` (with space). Use this exact key or modify `processing_mllama.py` to use `"pixel_values"`.

**Issue**: Out of memory during training
- **Solutions**:
  - Reduce batch size
  - Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
  - Use mixed precision training (FP16/BF16)
  - Apply LoRA instead of full fine-tuning

**Issue**: Slow inference
- **Solutions**:
  - Use KV-caching for autoregressive generation
  - Compile model with `torch.compile()`
  - Quantize model to INT8/INT4
  - Use smaller model variants

**Issue**: Model produces nonsensical outputs
- **Solutions**:
  - Ensure you're loading pretrained weights correctly
  - Check tokenizer vocabulary matches model config
  - Verify image preprocessing (normalization, resizing)
  - Adjust generation parameters (temperature, top_p, top_k)

### GPU Requirements

| Model Size | Minimum VRAM | Recommended VRAM | Batch Size |
|------------|--------------|------------------|------------|
| Small (1B) | 4 GB | 8 GB | 1-4 |
| Medium (3B) | 8 GB | 16 GB | 1-2 |
| Large (11B) | 16 GB | 24 GB | 1 |
| XLarge (90B) | 40 GB | 80 GB | 1 |

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this code in your research, please cite:

```bibtex
@software{llama32_multimodal,
  author = {Emmanuel Alo},
  title = {LLaMA 3.2 Multimodal Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/emmanuelalo52/LLaMA-3.2-Multimodal}
}
```

---

## 🙏 Acknowledgments

This implementation draws inspiration from:
- [Meta's LLaMA](https://github.com/meta-llama/llama) - Original LLaMA architecture
- [SigLIP](https://arxiv.org/abs/2303.15343) - Vision encoder design
- [LoRA](https://arxiv.org/abs/2106.09685) - Efficient fine-tuning method
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model implementations and utilities

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/emmanuelalo52/LLaMA-3.2-Multimodal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/emmanuelalo52/LLaMA-3.2-Multimodal/discussions)
- **Email**: Your contact email here

---

## 📊 Roadmap

- [ ] Add support for video input
- [ ] Implement distributed training utilities
- [ ] Add model quantization (INT8/INT4)
- [ ] Create Gradio/Streamlit demo
- [ ] Add more evaluation scripts
- [ ] Support for other vision encoders (CLIP, DinoV2)
- [ ] Integration with LangChain/LlamaIndex
- [ ] Docker containerization
- [ ] Model zoo with pretrained weights

---

<div align="center">

**Made with ❤️ by the open-source community**

⭐ Star this repo if you find it helpful!

</div>
