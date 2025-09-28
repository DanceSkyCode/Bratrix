import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("D:/fzh/15-EEG/Neural-MCRL-main")
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from EEGToVisual.datasets import EEGDataset, MEGDataset
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
import csv
from torch import Tensor
import itertools
import math
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


class Config:
    def __init__(self):
        self.task_name = 'classification'  
        self.seq_len = 201             
        self.pred_len = 201                
        self.output_attention = False      
        self.d_model = 201
        self.embed = 'timeF'               
        self.freq = 'h'                    
        self.dropout = 0.25                
        self.factor = 1                    
        self.n_heads = 4                   
        self.e_layers = 1                  
        self.d_ff = 256                    
        self.activation = 'gelu'         
        self.enc_in = 271

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=4):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :271, :]      
        return enc_out

class EnhancedNSAM(nn.Module):
    def __init__(self, num_channels: int = 271, seq_length: int = 256, sampling_rate: float = 250.0):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate

        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 55),
            'high_gamma': (55, 80)
        }

        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
            nn.Sigmoid()
        )

        self.spectral_attention = nn.Sequential(
            nn.Linear(len(self.bands), len(self.bands)),
            nn.GELU(),
            nn.Linear(len(self.bands), len(self.bands)),
            nn.Softmax(dim=-1)
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(seq_length)

    def get_band_mask(self, freqs: torch.Tensor, band: str) -> torch.Tensor:
        low, high = self.bands[band]
        return ((freqs >= low) & (freqs <= high)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, num_channels, seq_length]
        """
        identity = x
        batch_size = x.shape[0]
        
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(self.seq_length, 1/self.sampling_rate).to(x.device)

        band_features = {}
        band_powers = []
        
        for band in self.bands.keys():
            mask = self.get_band_mask(freqs, band).to(x.device)
            X_band = X * mask.unsqueeze(0).unsqueeze(0)
            band_features[band] = X_band
            power = torch.sum(torch.abs(X_band).pow(2), dim=-1) 
            band_powers.append(power)

        band_powers = torch.stack(band_powers, dim=-1)
        
        channel_weights = self.channel_attention(band_powers.mean(dim=-1))
        channel_weights = channel_weights.unsqueeze(-1) 

        spectral_input = band_powers.mean(dim=1)  
        spectral_weights = self.spectral_attention(spectral_input) 
        
        X_combined = torch.zeros_like(X)
        for i, band in enumerate(self.bands.keys()):
            X_combined += (band_features[band] * 
                         channel_weights * 
                         spectral_weights[:, i:i+1].unsqueeze(1))

        output = torch.fft.irfft(X_combined, n=self.seq_length, dim=-1)
        output = self.norm(output)
        
        alpha = torch.sigmoid(self.alpha)
        output = alpha * output + (1 - alpha) * identity
        
        return output
def topk_sparsify(x, k=64):
    topk_values, _ = torch.topk(x.abs(), k, dim=1)
    threshold = topk_values[:, -1].unsqueeze(1)
    return x * (x.abs() >= threshold)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=271, emb_size=40):
        super().__init__()
        self.in_channels = in_channels
        self.emb_size = emb_size

        # 初始通道压缩：271 -> 128（无残差，因通道数变化）
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU()

        # 残差块1：保持128通道，增加卷积深度（无下采样）
        self.res_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),  # 新增卷积层
            nn.BatchNorm2d(128),
        )
        self.elu_res1 = nn.ELU()

        # 残差块2：第一次下采样（时间维度减半）
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 11), stride=(1, 2), padding=(0, 5)),  # 下采样
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4)),  # 新增卷积层
            nn.BatchNorm2d(128),
        )
        self.elu_res2 = nn.ELU()
        # 下采样适配层（残差连接用，因stride=2导致时间维度变化）
        self.downsample2 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 2))

        # 残差块3：第二次下采样（时间维度再减半）
        self.res_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7)),  # 下采样
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 13), stride=(1, 1), padding=(0, 6)),  # 新增卷积层
            nn.BatchNorm2d(128),
        )
        self.elu_res3 = nn.ELU()
        # 下采样适配层
        self.downsample3 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 2))

        # 残差块4：空间卷积（融合通道特征）
        self.res_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),  # 新增卷积层
            nn.BatchNorm2d(128),
        )
        self.elu_res4 = nn.ELU()

        # 通道压缩到目标维度：128 -> 40（无残差）
        self.conv_final = nn.Conv2d(128, emb_size, kernel_size=(1, 1))
        self.bn_final = nn.BatchNorm2d(emb_size)
        self.elu_final = nn.ELU()

        # 自适应池化固定时间维度为36
        self.pool = nn.AdaptiveAvgPool2d((1, 36))
        self.rearrange = Rearrange('b c 1 w -> b c w')

        # 验证第一层卷积权重形状
        assert self.conv1.weight.shape == (128, in_channels, 1, 1), \
            f"卷积层1权重形状错误！期望(128, {in_channels}, 1, 1)，实际{self.conv1.weight.shape}"

    def forward(self, x):
        # 输入形状：[B, 271, 201]
        assert x.shape[1] == self.in_channels, \
            f"输入通道数错误！期望{self.in_channels}，实际{x.shape[1]}"

        # 增加空间维度：[B, 271, 1, 201]
        x = x.unsqueeze(2)

        # 初始通道压缩
        x = self.elu1(self.bn1(self.conv1(x)))  # [B, 128, 1, 201]

        # 残差块1（无下采样）
        residual = x  # 跳跃连接：保留原始输入
        x = self.res_block1(x)  # 经过两层卷积
        x += residual  # 残差相加
        x = self.elu_res1(x)  # [B, 128, 1, 201]（形状不变）

        # 残差块2（第一次下采样）
        residual = self.downsample2(x)  # 下采样适配：[B, 128, 1, 101]
        x = self.res_block2(x)  # 经过两层卷积（含下采样）
        x += residual  # 残差相加
        x = self.elu_res2(x)  # [B, 128, 1, 101]

        # 残差块3（第二次下采样）
        residual = self.downsample3(x)  # 下采样适配：[B, 128, 1, 51]
        x = self.res_block3(x)  # 经过两层卷积（含下采样）
        x += residual  # 残差相加
        x = self.elu_res3(x)  # [B, 128, 1, 51]

        # 残差块4（空间卷积）
        residual = x  # 跳跃连接
        x = self.res_block4(x)  # 经过两层空间卷积
        x += residual  # 残差相加
        x = self.elu_res4(x)  # [B, 128, 1, 51]

        # 通道压缩到目标维度
        x = self.elu_final(self.bn_final(self.conv_final(x)))  # [B, 40, 1, 51]

        # 固定时间维度为36
        x = self.pool(x)  # [B, 40, 1, 36]

        # 移除多余维度
        x = self.rearrange(x)  # [B, 40, 36]

        return x
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects + 1, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5
        
    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)
        
    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, num_channels=271, seq_length=201, d_model=201, num_scales=5):
        super().__init__(
            PatchEmbedding(emb_size=40),
            FlattenHead()
        ) 
     
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)
class LinearFusion(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1)  # 学习每个位置的权重
        
    def forward(self, x):
        # x: [200, 5, 1024]
        weights = torch.softmax(self.attention(x).squeeze(2), dim=1)  # 权重：[200, 5]
        weights = weights.unsqueeze(2)  # 扩展维度：[200, 5, 1]
        x_fused = (x * weights).sum(dim=1)  # 加权求和：[200, 1024]
        return x_fused


class InterMCRAlignment(nn.Module):
    def __init__(self, d_model=201, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # self.text_eeg_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        
        # self.text_image_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # self.norm2 = nn.LayerNorm(d_model)
        
        # self.final_alignment = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # self.norm3 = nn.LayerNorm(d_model)
        
        # self.eeg_proj = nn.Linear(250, d_model)
        self.image_proj = nn.Linear(1024, 1024)
        self.text_proj = nn.Linear(1024, 1024)
        
        # self.output_proj = nn.Linear(d_model, 250)
        
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        self.sparse_encoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5)  # 强制稀疏化# 输出稀疏编码长度 k
        )

        self.sparse_decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024*1024),# 输出稀疏编码长度 k
            nn.Dropout(0.5)  # 强制稀疏化
        )
        self.image_uncertainty_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),          # 输出一个不确定性标量
            nn.Softplus()              # 保证输出为正值，避免负不确定性
        )
        self.text_uncertainty_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),          # 输出一个不确定性标量
            nn.Softplus()              # 保证输出为正值，避免负不确定性
        )
        self.fusion_img = LinearFusion()
        self.fusion_text = LinearFusion()
        self.proj_eeg = Proj_eeg(embedding_dim=1024)
        self.prior_matrix_mri = nn.Parameter(torch.zeros(1024, 1024))
        self.prior_matrix_img = nn.Parameter(torch.zeros(1024, 1024))
    def uncertainty(self, image_features, text_features):
        image_logits = self.image_uncertainty_head(image_features)  # [B, 4, K]
        text_logits  = self.text_uncertainty_head(text_features)    # [B, 4, K]

        # 证据：exp(logit) 保证非负；+1 得到 Dirichlet 参数 α
        image_evidence = torch.exp(image_logits)
        text_evidence = torch.exp(text_logits)

        image_alpha = image_evidence + 1  # [B, 4, K]
        text_alpha  = text_evidence + 1


        K = image_alpha.shape[-1]
        image_S = image_alpha.sum(dim=-1)  # [B, 4]
        text_S  = text_alpha.sum(dim=-1)   # [B, 4]

        image_u = K / image_S  # [B, 4]
        text_u  = K / text_S
        # 权重为 (1 - u): 可靠性高的视图权重大
        image_weights = 1.0 - image_u  # [B, 4]
        text_weights  = 1.0 - text_u   # [B, 4]

        # 加权求和：先扩展维度便于广播
        image_weighted = (image_features * image_weights.unsqueeze(-1)).sum(dim=1)  # [B, 1024]
        text_weighted  = (text_features  * text_weights.unsqueeze(-1)).sum(dim=1)

        # 归一化
        image_weights_sum = image_weights.sum(dim=1, keepdim=True) + 1e-8
        text_weights_sum  = text_weights.sum(dim=1, keepdim=True) + 1e-8

        image_fused = image_weighted / image_weights_sum  # [B, 1024]
        text_fused  = text_weighted  / text_weights_sum
        image_proj = self.image_proj(image_fused)  # [B, 1024]
        text_proj  = self.text_proj(text_fused)   # [B, 1024]
        return image_proj + self.fusion_img(image_features), text_proj + self.fusion_text(text_features)

    def matrix(self, image_proj, text_proj):
        sparse_code_img = self.sparse_encoder(image_proj)           # [B, k]
        weight_matrix = self.sparse_decoder(sparse_code_img)  
        weight_matrix_image = weight_matrix.view(-1, 1024, 1024)  + self.prior_matrix_img # [B, 1024, 1024]
        weight_matrix_image = torch.sigmoid(weight_matrix_image)  # 保持在 0~1，激活稀疏区域
        if self.training:
            image_text_feature = torch.bmm(image_proj.unsqueeze(2), text_proj.unsqueeze(1))  # outer product
            weighted_image_text = image_text_feature * weight_matrix_image
            pooled_image_feature = weighted_image_text.mean(dim=2)  # [B, 1024]
        else:
            pooled_image_feature = image_proj * (weight_matrix_image.mean(dim=2))
        return pooled_image_feature, sparse_code_img

    def forward(self, eeg_features_o, image_features, text_features, epoch, round_gap=8):
        B, D = eeg_features_o.size()
        image_proj, text_proj = self.uncertainty(image_features, text_features)
        eeg_features = self.proj_eeg(eeg_features_o) # torch.Size([256, 1024])
        # 1. 计算 eeg_text 特征交互矩阵

        # 2. 生成随机 mask，控制混合
        if self.training:
            eeg_text_feature = torch.bmm(eeg_features_o.unsqueeze(2), text_proj.unsqueeze(1))  # [B, 1024, 1024]
            # lambda_ratio = min(0.8, 0.1 + 0.7 * int(epoch/40))  # 持续增长，最多 0.8
            # mask = (torch.rand(B, D, device=eeg_features.device) < lambda_ratio).float()  # [B, 1024]

            # if epoch % round_gap == 0:
            #     # 模态1主导
            #     mixed_token = image_proj * mask + eeg_features * (1 - mask)  # [B, 1024]
            # elif epoch % round_gap == round_gap - 1:
            #     # 模态2主导
            #     mixed_token = eeg_features * mask + image_proj * (1 - mask)  # [B, 1024]
            # else:
            #     # 不混合，正常训练
            #     mixed_token = eeg_features

        # 3. 稀疏编码：从混合token生成稀疏权重矩阵
            sparse_code_eeg = self.sparse_encoder(eeg_features)           # [B, k]
            # sparse_code_eeg = topk_sparsify(sparse_code_eeg)  

            weight_matrix = self.sparse_decoder(sparse_code_eeg)         # [B, 1024 * 1024]
            weight_matrix_eeg = weight_matrix.view(-1, 1024, 1024) + self.prior_matrix_mri   # [B, 1024, 1024]
            weight_matrix_eeg = torch.sigmoid(weight_matrix_eeg)
            # eeg_features = eeg_features * weight_matrix_eeg.mean(dim=2)
            weighted_eeg_text = eeg_text_feature * weight_matrix_eeg     # [B, 1024, 1024]
            pooled_eeg_feature = weighted_eeg_text.mean(dim=2)       # [B, 1024]
            pooled_image_feature, sparse_code_img = self.matrix(image_proj, text_proj)
            # image_proj = image_proj * weight_matrix_eeg.mean(dim=2)

            # eps = 1e-8
            p = F.softmax(sparse_code_eeg, dim=-1)
            q = F.softmax(sparse_code_img, dim=-1)
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(p.log(), q, reduction='batchmean')

            # 2. 稀疏权重矩阵一致性损失（L2 或 Cosine）
            # weight_consistency_loss = F.mse_loss(sparse_code_eeg.detach(), sparse_code_img.detach()) *0.3
            weight_consistency_loss = 0
            
            return pooled_eeg_feature, pooled_image_feature, kl_loss, weight_consistency_loss, image_proj, eeg_features_o
        else:
            sparse_code_eeg = self.sparse_encoder(eeg_features)           # [B, k]
            weight_matrix = self.sparse_decoder(sparse_code_eeg)         # [B, 1024 * 1024]

            weight_matrix_eeg = weight_matrix.view(-1, 1024, 1024)   + self.prior_matrix_mri    # [B, 1024, 1024]
            weight_matrix_eeg = torch.sigmoid(weight_matrix_eeg)
            weight_matrix_eeg = (eeg_features * weight_matrix_eeg).mean(dim=2)
            eeg_features_o = torch.cat((eeg_features_o, weight_matrix_eeg),dim = 1)
            sparse_code_img = self.sparse_encoder(image_proj)           # [B, k]
            weight_matrix = self.sparse_decoder(sparse_code_img)  
            weight_matrix_eeg = weight_matrix.view(-1, 1024, 1024) + self.prior_matrix_img      # [B, 1024, 1024]
            weight_matrix_eeg = torch.sigmoid(weight_matrix_eeg)
            weight_matrix_eeg = (image_proj * weight_matrix_eeg).mean(dim=2)
            image_proj = torch.cat((image_proj, weight_matrix_eeg),dim = 1)
            return image_proj, eeg_features_o
def load_pretrained_inter_mcr(model, ckpt_path):
    """加载预训练参数，并且 key 以 inter_mcr. 开头"""
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 取出 state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 过滤出 inter_mcr. 开头的 key
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("inter_mcr."):
            new_state_dict[k[len("inter_mcr."):]] = v  # 去掉前缀以匹配 model

    # 加载参数
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model

class NoiseAugmentation(nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_eeg, z_image, batch_size):
        """
        z_eeg: [N, D] normalized
        z_image: [N, D] normalized
        """
        logits = torch.matmul(z_eeg, z_image.T) / self.temperature
        
        labels = torch.arange(batch_size, device=z_eeg.device)
    
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2.0


class NeuralMCRL(nn.Module):    
    def __init__(self, num_channels=271, sequence_length=201, num_subjects=4, num_features=64, num_latents=1024, num_blocks=1):
        super(NeuralMCRL, self).__init__()
        default_config = Config()
        d_model = 201
        
        self.subject_layer = SubjectLayers(
            in_channels=num_channels,
            out_channels=num_channels,
            n_subjects=num_subjects,
            init_id=True
        )
        self.encoder = iTransformer(default_config)
        # self.nsam = EnhancedNSAM(
        #     num_channels=num_channels,
        #     seq_length=201,
        #     sampling_rate=250.0
        # )
        
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        
        self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
        
        self.inter_mcr = InterMCRAlignment(
            d_model=d_model,
            num_heads=3,
            dropout=default_config.dropout
        )
        # path = r"D:\fzh\15-EEG\Neural-MCRL-main\models\contrast\across\NeuralMCRL-eeg-sub-01-08-19_20-03\best_top5-0.4600.pth"
        # self.inter_mcr = load_pretrained_inter_mcr(self.inter_mcr, path)
        self.noise_aug = NoiseAugmentation(sigma=0.01)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids, text_features=None, img_features=None, epoch=None):
        x = self.subject_layer(x, subject_ids) # torch.Size([256, 63, 250]) torch.Size([256])
        x_trans = self.encoder(x, None, subject_ids) # torch.Size([256, 63, 250])
        # x_processed = self.nsam(x) # torch.Size([256, 63, 250])
        x_normalized = self.feature_norm(x_trans)
        eeg_features = self.enc_eeg(x_normalized) # torch.Size([256, 1440])
        eeg_projected = self.proj_eeg(eeg_features) # torch.Size([256, 1024])

        
        if self.training:
            x_aligned, pooled_image_feature, kl_loss, weight_consistency_loss, img_features, eeg_features = self.inter_mcr(eeg_projected, img_features, text_features, epoch)
            final_features = x_aligned + eeg_projected
        else:
            img_features, eeg_features = self.inter_mcr(eeg_projected, img_features, text_features, epoch)
        if self.training:
            final_features = self.noise_aug(final_features)
            return final_features, pooled_image_feature, img_features, eeg_features, kl_loss, weight_consistency_loss
        else:
            return  eeg_features, img_features
 
def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train_model(sub, eeg_model, dataloader, optimizer, scheduler, device, text_features_all, img_features_all, config, epoch):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    lamda1 = 0.2
    lamda2 = 0.2
    lamda3 = 0.5
    features_list = []
    save_features= True
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = eeg_data.size(0)  
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)  
        pooled_eeg_feature, pooled_image_feature, img_features, eeg_features, kl_loss, weight_consistency_loss = eeg_model(eeg_data, subject_ids, text_features, img_features, epoch)

        
        features_list.append(pooled_eeg_feature.float())
        logit_scale = eeg_model.logit_scale
        
        pool_img_loss = eeg_model.loss_func(pooled_eeg_feature, pooled_image_feature, logit_scale)
        img_loss = eeg_model.loss_func(img_features, eeg_features, logit_scale)
        loss =  lamda3 * pool_img_loss + img_loss + lamda1 * kl_loss + lamda2 * weight_consistency_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
         
        logits_img = logit_scale * eeg_features @ img_features.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)  

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features, pooled_eeg_feature, pooled_image_feature
    if epoch == 20 :
        current_lr = optimizer.param_groups[0]['lr'] * 0.1
        print(f"Epoch {epoch}, 当前学习率: {current_lr:.6f}")
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, config, epoch, k):
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0

    save_path = 'D:/fzh/15-EEG/Neural-MCRL-main/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_eeg_features = []
    all_top5_indices = []
    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0) 
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)       
            eeg_features, img_features = eeg_model(eeg_data, subject_ids, text_features, img_features, epoch)
            img_features_all_de, text_features_all_de = eeg_model.inter_mcr.uncertainty(img_features_all, text_features_all)
            img_features_all_de_2, _ = eeg_model.inter_mcr.matrix(img_features_all_de, text_features_all_de)
            img_features_all_de = torch.cat((img_features_all_de, img_features_all_de_2), dim=1)
            all_eeg_features.append(eeg_features.cpu().numpy())
        
            logit_scale = eeg_model.logit_scale
            loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)

            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all_de[selected_classes]
                # selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():

                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                    all_top5_indices.append([selected_classes[i] for i in top5_indices.tolist()])
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features

    all_eeg_features = np.vstack(all_eeg_features)  
    np.save(os.path.join(save_path, f'eeg_features_{sub}_epoch{epoch}.npy'), all_eeg_features) 
    
    top5_df = pd.DataFrame(all_top5_indices, columns=[f'Top5_Idx_{i+1}' for i in range(5)])  
    top5_df.to_csv(os.path.join(save_path, f'top5_indices_{sub}_epoch{epoch}.csv'), index=False) 
    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc



def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, scheduler, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  
    
    best_top5 = 0  # 初始化记录最佳 top-5 准确率

    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy, features_tensor = train_model(
            sub, eeg_model, train_dataloader, optimizer, scheduler, device, 
            text_features_train_all, img_features_train_all, config=config, epoch=epoch
        )

        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(
            sub, eeg_model, test_dataloader, device, 
            text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=200
        )
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, 
                                    text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=2)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, 
                                    text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=4)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, 
                                    text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=10)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, 
                                                text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=50)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, 
                                                    text_features_test_all, img_features_test_all, config=config, epoch=epoch, k=100)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        # 保存 top-5 最佳模型
        if top5_acc > best_top5 and epoch > 20:
            best_top5 = top5_acc
            if config.insubject:
                save_dir = f"./models/contrast/{config.encoder_type}-meg-{sub}-{current_time}"
            else:
                save_dir = f"./models/contrast/across/{config.encoder_type}-meg-{sub}-{current_time}"
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)

            # 删除旧模型（如果存在）
            old_model_path = os.path.join(save_dir, "best_top5.pth")
            if os.path.exists(old_model_path):
                os.remove(old_model_path)

            # 新模型路径，文件名里加 best_top5 精度
            file_path = os.path.join(save_dir, f"best_top5-{best_top5:.4f}.pth")
            torch.save(eeg_model.state_dict(), file_path)

            print(f"New best top-5 model saved! Top-5 acc: {best_top5:.4f}, path: {file_path}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
         
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
             
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
     
    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

import datetime

def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="C:/fzh/MEG/THINGSMEG/THINGS-MEG/things-meg/Preprocessed_data", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='NeuralMCRL', help='Encoder type') 
    parser.add_argument('--subjects', nargs='+', default=['sub-01','sub-02','sub-03', 'sub-04'], help='List of subject IDs (default: sub-01 to sub-10)')   
    args = parser.parse_args()

      
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects        
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,  # 每隔50轮调整一次
            gamma=0.1      # 调整倍数：新学习率 = 原学习率 × 0.1
        )
        if args.insubject:
            train_dataset = MEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = MEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = MEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = MEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, scheduler, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config=args, logger=args.logger)

 
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

                
if __name__ == '__main__':
    main()
