import torch
import numpy as np
from PIL import Image

IMAGENET_STANDARD_MEAN = [
    0.48145466,
    0.4578275,
    0.40821073
  ]
IMAGENET_STANDARD_STD = [
    0.26862954,
    0.26130258,
    0.27577711
  ]

def add_image_tokens_to_prompts(prefix_prompt,bos_token,image_seq_len,image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(image,scale,dtype=np.float32):
    rescaled_image = image*scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image,mean,std):
    mean = np.array(mean,dtype=image.dtype)
    std = np.array(std,dtype=image.dtype)
    image = (image - mean)/std
    return image

def resize(image,size,resample=None,reducing_gap=None):
    height,width = size
    resized_image = image.resize((width,height),resample=resample,reducing_gap=reducing_gap)
    return resized_image

def process_images(images,size=None,resample=None,rescale_factor=None,image_mean=None,image_std=None):
    height,width = size[0],size[1]
    images = [resize(image=image,size=(height,width),resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image,scale=rescale_factor) for image in images]
    images = [normalize(image,mean=image_mean,std=image_std) for image in images]
    #Rescale it to [channel, height, width]
    images = [image.transpose(2,0,1) for image in images]
    return images

class MllamaImageProcessor:
    IMAGE_TOKEN = "<image>"
    def __init__(self,tokenizer,num_image_token,image_size):
        super().__init__()
        self.image_seq_length = num_image_token
        self.image_size = image_size

        tokens_to_add = {"additional_special tokens":[self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)] # These are object detection tokens
        EXTRA_TOKENS = [f"<seg{i:03d}>" for i in range(128)] # For object segmentation tokens
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # ADD BOS and EOS 
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
    def __call__(self,text,images,padding,truncation=True):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"
        pixel_values = process_images(
            images,
            size = (self.image_size,self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        pixel_values = np.stack(pixel_values,axis=0) #Convert to single tensor
        pixel_values = torch.tensor(pixel_values)

        # Create place holder for CLIP and prompts
        input_strings = [
            add_image_tokens_to_prompts(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(input_strings,return_tensors = "pt",padding=padding,truncation=truncation)

        return_data = {"pixel Value": pixel_values, **inputs}

        return return_data