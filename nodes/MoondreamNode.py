import os,sys
import folder_paths
from PIL import Image
import numpy as np

import torch
from comfy.model_management import get_torch_device
from huggingface_hub import hf_hub_download

# print('#######s',os.path.join(__file__,'../'))

# sys.path.append(os.path.join(__file__,'../../'))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

                
# from moondream import VisionEncoder, TextModel,detect_device
# from transformers import TextIteratorStreamer

from moondream import Moondream, detect_device
from threading import Thread
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer



# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class MoondreamNode:
    def __init__(self):
        # get_torch_device()
        self.moondream = None
        self.tokenizer = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "question": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '',
                            "dynamicPrompts": False
                          }),
            "moondream_model": ("MOONDREAM_MODEL",),
                             },
                } 
    
    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Prompt"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
  
    def run(self, image, question, moondream_model):
        result = []
        self.moondream = moondream_model[0][0]
        self.tokenizer = moondream_model[0][1]

        # Process each image with the provided question
        question = question[0]  # Assuming single question for all images
        for im in image:
            im = tensor2pil(im)  # Convert tensor image to PIL for processing
            image_embeds = self.moondream.encode_image(im)
            res = self.moondream.answer_question(image_embeds, question, self.tokenizer)
            result.append(res)

        return (result,),

class LoadMoondreamModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu","cuda"],),
            },
        }

    RETURN_TYPES = ("MOONDREAM_MODEL",)
    FUNCTION = "load_moondream_model"
    CATEGORY = "♾️Mixlab/Prompt"

    def load_moondream_model(self, device):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoints_dir = os.path.join(current_dir, '..', 'checkpoints')

        # Ensure the checkpoints directory exists
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)

        # Define the files to download
        files_to_download = {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "tokenizer.json": "tokenizer.json",  # Add tokenizer here to avoid redundant code
        }

        # Download the files if they do not exist
        for local_filename, hf_filename in files_to_download.items():
            local_file_path = os.path.join(checkpoints_dir, local_filename)
            if not os.path.exists(local_file_path):
                hf_hub_download(
                    repo_id="vikhyatk/moondream1",
                    filename=hf_filename,
                    local_dir=checkpoints_dir,
                    endpoint='https://hf-mirror.com'
                )

        if device != self.device.type:
            self.device = torch.device(device)
        
        # Load the Moondream model
        moondream_model = Moondream.from_pretrained(checkpoints_dir).to(device=self.device, dtype=self.dtype)
        moondream_model.eval()  # Set the model to evaluation mode
        
        # Load the tokenizer
        tokenizer = Tokenizer.from_pretrained(checkpoints_dir)

        return ((moondream_model,tokenizer),)
