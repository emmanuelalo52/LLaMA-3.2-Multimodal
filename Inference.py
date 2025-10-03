from PIL import Image
import torch
import fire

from processing_mllama import MllamaImageProcessor
from model import KVCache, MllamaForConditionalGeneration
from utils import load_hf_model