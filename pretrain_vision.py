from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import torch
from utils import *
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torch import nn
from itertools import chain
from copy import deepcopy


FOLDER = "ctr_images"

pretrained_name = "Qwen/Qwen2.5-VL-7B-Instruct"
vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_name,
    dtype = torch.bfloat16,
    trust_remote_code = True
).model.visual
vision_model.to(torch.device("cuda"))
processor = Qwen2_5_VLProcessor.from_pretrained(
    pretrained_name, 
    trust_remote_code = True
)


lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = "all-linear",
    bias = "none",
)
vision_model = get_peft_model(vision_model, lora_config)


class CtrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vision_model
        self.proj = nn.Sequential(
            nn.Linear(3584, 1792),
            nn.SiLU(),
            nn.Linear(1792, 3584),
        )
        
    def forward(self, **kwargs):
        emb = self.encoder(**kwargs)
        return self.proj(emb)


class ImgDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = [f"{FOLDER}/{p}" for p in os.listdir(FOLDER)]
        self.transform = transforms.Compose([
            transforms.ColorJitter(
                brightness = 0.1,
                contrast = 0.1,
            )
        ])
        
    def __len__(self):
        return len(self.paths)
        
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


def get_linear_schedule_with_end(optimizer, num_training_steps, lr_start, lr_end):
    def lr_lambda(current_step):
        if current_step >= num_training_steps:
            return lr_end / lr_start
        progress = current_step / num_training_steps
        return (1 - progress) * (1 - lr_end / lr_start) + lr_end / lr_start

    return LambdaLR(optimizer, lr_lambda)


epoch = 5
batch_size = 512
logs_step = 10

train_ds = ImgDataset()
train_dl = get_dataloader(train_ds, batch_size = batch_size, shuffle = True)
repeated_train_dl = chain.from_iterable([train_dl] * epoch)
model = CtrModel().to(torch.device("cuda"))
optimizer = AdamW(model.parameters(), lr = 5e-4)

pbar = tqdm(repeated_train_dl, total = len(train_dl) * epoch, ncols = 100)
his = []
best_state_dict = None

for step, batch in enumerate(pbar, 1):
    emb = model(**batch)
    
    loss = contrastive_loss(emb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if loss.item() < min(his):
        best_state_dict = deepcopy(model.state_dict())
    
    his.append(loss.item())
    
    if step % logs_step == 0:
        print(f"Step: {step}, loss: {np.mean(his[:-logs_step:])}")
    
    pbar.set_postfix(loss = loss.item())
    
torch.save(best_state_dict, "pretrained_vision.torch")