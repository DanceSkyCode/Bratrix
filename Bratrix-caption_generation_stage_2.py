import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import open_clip, time
import seaborn as sns
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("")
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from NeuralMCRL_revise_image_caption import BrainXC
from EEGToVisual.datasets import EEGDataset
from matplotlib.colors import LinearSegmentedColormap
from transformers import LlamaForCausalLM, LlamaTokenizer
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
import random
from transformers import CLIPVisionModel
from perceiver import PerceiverResampler 
from util import wandb_logger
import csv
from torch import Tensor
import itertools
import math
from custom_pipeline import *
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW
import pandas as pd 
print("import ok!")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')
save_path = "Bratrix"
sub="sub-01"
epoch = "0"
all_eeg_features_test = np.load(os.path.join(save_path, f'eeg_features_{sub}_epoch{epoch}_test.npy'))
all_eeg_features_test = torch.tensor(all_eeg_features_test).to(device).float()
all_eeg_features_test = all_eeg_features_test[:,:1024]
class LazyEEGImageDataset(Dataset):
    def __init__(self, save_path, sub, epoch, split="train", device="cpu"):
        self.save_path = save_path
        self.sub = sub
        self.epoch = epoch
        self.split = split
        self.device = device
        self.eeg_features = np.load(
            os.path.join(save_path, f"eeg_features_{sub}_epoch{epoch}_{split}.npy"),
            mmap_mode='r'
        )[:, :1024]  
        self.img_features = np.load(
            os.path.join(save_path, f"img_features_all_de_{sub}_epoch{epoch}_{split}.npy"),
            mmap_mode='r'
        )

    def __len__(self):
        return len(self.eeg_features)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg_features[idx]).float()
        img = torch.from_numpy(self.img_features[idx]).float()
        return eeg, img
dataset = LazyEEGImageDataset(save_path, sub, epoch, split="train", device=device)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=85, shuffle=True, num_workers=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_path = "Bratrix/checkpints/mm_projector.bin"
num_epochs = 240
mm_projector = nn.Linear(1024, 4096)
mm_projector_weights = torch.load(weight_path, map_location='cpu')
mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
mm_projector.to(device)
voxel2emb = BrainXC().to(device)
model_path = r"Bratrix/checkpoints/voxel2emb.pth"
checkpoint = torch.load(model_path, map_location=device)
model_state_dict = checkpoint["model_state_dict"]
voxel2emb.load_state_dict(model_state_dict)
optimizer = torch.optim.Adam(list(mm_projector.parameters()) + list(voxel2emb.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# for epoch in range(num_epochs):
#     mm_projector.train()
#     # mm_projector.eval() 
#     voxel2emb.train()
#     epoch_loss = 0.0

#     for train_i, (eeg, image) in enumerate(dataloader):
#         optimizer.zero_grad()
#         eeg = eeg.to(device)
#         image = image.to(device)

#         # with torch.no_grad():   # mm_projector 不需要梯度
#         image_proj = mm_projector(image)
#         eeg_emb = voxel2emb(eeg)

#         loss_mse = criterion(eeg_emb, image_proj)
#         loss_mse.backward()
#         optimizer.step()

#         epoch_loss += loss_mse.item()

#     avg_loss = epoch_loss / len(dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] - Avg MSE Loss: {avg_loss:.6f}")

#     save_dir = "./checkpoints"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"voxel2emb_epoch{epoch}.pth")

#     torch.save({
#         'epoch': num_epochs,
#         'model_state_dict': voxel2emb.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': avg_loss,
#     }, save_path)

save_path = "Bratrix/output/output.txt"
num_samples = 200

finetuned_llama = 'shikra-model' # 'model_weights/shikra-7b' # shikra
tokenizer = LlamaTokenizer.from_pretrained(finetuned_llama, padding_side='left')
model = LlamaForCausalLM.from_pretrained(finetuned_llama, torch_dtype=torch.bfloat16)
model.to(device)
gen_kwargs = dict(
    use_cache=True,
    do_sample=False,
    pad_token_id=2, # tokenizer.pad_token_id,
    bos_token_id=1, # tokenizer.bos_token_id,
    eos_token_id=2, # tokenizer.eos_token_id,
    max_new_tokens=512,
)
system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
base_prompt = "Describe this image <image> as simply as possible."
num_patches = 256

with open(save_path, 'w', encoding='utf-8') as f:
    for k in range(num_samples):
        user_image_placeholder = "<im_start>" + "<im_patch>" * num_patches + "<im_end>"
        tokens = tokenizer.tokenize(user_image_placeholder)
        patch_count = tokens.count("<im_patch>")
        if patch_count != num_patches:
            print(f"warning: in fact {patch_count}个<im_patch>, requied {num_patches}个")
        if "<image>" in base_prompt:
            user_prompt = base_prompt.replace("<image>", user_image_placeholder)
        else:
            user_prompt = base_prompt + user_image_placeholder
        input_text = system_prompt + user_prompt + " ASSISTANT:"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        inputs_embeds = model.model.embed_tokens(input_ids)
        eeg_embeds = all_eeg_features_test[k:k+1].to(device)  # [1, 256, 1024]
        eeg_embeds = voxel2emb(eeg_embeds)
        image_start_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
        image_end_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            cur_new_embeds = cur_input_embeds.clone()
            start_pos = (cur_input_ids == image_start_token_id).nonzero(as_tuple=False)[0].item()
            end_pos = (cur_input_ids == image_end_token_id).nonzero(as_tuple=False)[0].item()

            if (end_pos - start_pos - 1) != num_patches:
                raise ValueError(f"Expected {num_patches} patches, found {end_pos - start_pos - 1}")
            cur_new_embeds[start_pos+1:end_pos, :] = eeg_embeds[0]
            new_input_embeds.append(cur_new_embeds)
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        st_time = time.time()
        with torch.inference_mode():
            with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
                output_ids = model.generate(inputs_embeds=inputs_embeds, **gen_kwargs)
        elapsed = time.time() - st_time

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"[{k+1}/{num_samples}] Generated in {elapsed:.2f}s: {response}")
        f.write(response + "\n")