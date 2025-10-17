from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import torch
from utils import *
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from peft import LoraConfig, get_peft_model


FOLDER = "ctr_images"

pretrained_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_name,
    dtype = torch.bfloat16,
    trust_remote_code = True
).model.visual
model.to(torch.device("cuda"))
processor = Qwen2_5_VLProcessor.from_pretrained(
    pretrained_name, 
    trust_remote_code = True
)


lora_config = LoraConfig(
    r = 8,
    alpha = 16,
    target_modules = "all_linear",
    bias = "none",
)
model = get_peft_model(model, lora_config)
print(model.get_nb_trainable_parameters())


class ImgDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = [f"{FOLDER}/{p}" for p in os.listdir(FOLDER)]
        self.transform = transforms.Compose([
            transforms.ColorJitter(
                brightness = 0.05,
                contrast = 0.05,
            )
        ])
        
    def __getitem__(self, index):
        batch = processor(
            text = [""],
            images = [Image.open(self.paths[index])]
        )
        renamed_batch = {
            "hidden_states": batch["pixel_values"],
            "grid_thw": batch["image_grid_thw"],
        }
        renamed_batch = {k: v.squeeze(0) for k, v in renamed_batch.items()}
        
        return renamed_batch
        
        

def contrastive_loss(embeddings, temperature = 1.0):
    embeddings = F.normalize(embeddings, p = 2, dim = -1)  # (N, D)
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

    labels = torch.eye(sim_matrix.shape[0], dtype = torch.int, device = embeddings.device)  # (N, N)

    logits = sim_matrix / temperature

    loss = F.cross_entropy(logits, labels)

    return loss


