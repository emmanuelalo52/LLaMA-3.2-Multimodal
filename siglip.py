from typing import Optional,Tuple
import torch
import torch.nn as nn

#Siglip model configuration
class SiglipVisionConfig:
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=12,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_channels=3,
                 image_size=224,
                 patch_size=16,
                 layer_norm_eps=1e-6,
                 attention_dropout=0.0,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

#Embedding layer
class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         padding="valid",)
        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1,-1)),
                             persistent=False,)
    
    def forward(self,pixel_values:torch.FloatTensor):
        _,_,height,width = pixel_values.shape
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1,2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class SiglipMLP(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)
    def forward(self,hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = nn.functional.gelu(hidden_state,approximate="tanh")
        hidden_state = self.fc2(hidden_state)
        return hidden_state
    
class SiglipAttention(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        """NOTE: We will be using Mutilhead attention. For this we will run the attention weights across mutiple heads and concatinate it into proceeding context vector"""
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** 0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim)

    def forward(self,hidden_states):
        batch_size, seq_len,_ = hidden_states.shape
        keys_state = self.k_proj.view(hidden_states)
        query_state = self.q_proj(hidden_states)
        value_state = self.v_proj(hidden_states)

        keys_state = keys_state.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        value_state = value_state.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        query_state = query_state.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        attention_weights = (torch.matmul(query_state, keys_state.transpose(2,3))*self.scale)
        if attention_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attention_weights.size()}"
            )
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attention_weights = nn.functional.softmax(attention_weights,dim=-1,dtype=torch.float32).to(query_state.dtype)
        attention_weights = nn.functional.dropout(attention_weights,p=self.dropout,training=self.training)
        context_vector = torch.matmul(attention_weights,value_state)
        if context_vector.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`context vector` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {context_vector.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        context_vector = context_vector.transpose(1,2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        context_vector = context_vector.reshape(batch_size,seq_len,self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        context_vector = self.out_proj(context_vector)
        return context_vector, attention_weights

# Encoder
class SiglipVisionEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.layernorm1 = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layernorm2 = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
    def forward(self,hidden_state):
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_state
        hidden_state = self.layernorm1(hidden_state)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_state = self.self_attn(hidden_state)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_state  = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layernorm2(hidden_state)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_state = self.mlp(hidden_state)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_state = residual + hidden_state
        
        return hidden_state
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipVisionEncoder(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(self, inputs_embeds):
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states
class SiglipVisionTransformer(nn.Module):
    """Explanation:
                input pixels
                      |
                      v
                Embedding Patch
                      |
                      v
               Positional Encoding
                      |
                      v
                Transformer Layer
                      |
                      v
             Contextualized Embedding"""
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.embedding = SiglipVisionEmbedding(config)
        self.encoder = SiglipVisionEncoder(config)

        self.layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
    def forward(self,pixel_values:torch.Tensor):
        #dimension [Batch_size,channels,height,width] -> dimension[Batch_size,num_patches,Embed_dim]
        hidden_states = self.embedding
        last_hidden_state = self.encoder(input=hidden_states)
        last_hidden_state = self.layernorm(last_hidden_state)

        return last_hidden_state
#Siglip Model
class SiglipModel(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    def forward(self,pixel_values):
        #dimension [Batch_size,channels,height,width] -> dimension[Batch_size,num_patches,Embed_dim]
        return self.vision_model(pixel_values=pixel_values)