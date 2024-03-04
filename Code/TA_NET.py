import os
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):   
        return self.resblocks(x)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):   
        return self.resblocks(x)

class TA_NET(nn.Module):
    def __init__(self, num_bags: int, bag_idx: int, width: int, layers: int, heads: int, layers_1: int, heads_1: int):
        super().__init__()
        self.num_bags = num_bags
        self.bag_idx = bag_idx
        
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        
        self.ln_pre_1 = nn.LayerNorm(width)
        self.transformer_1 = Transformer(width, layers_1, heads_1)
        self.ln_post_1 = nn.LayerNorm(width)
        self.fc = nn.Sequential(
            nn.Linear(width, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def generate_time_sequence(self, tokens):
        tokens = torch.cat([torch.zeros(self.bag_idx-1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device), tokens],dim=0)
        tokens = torch.cat([tokens, torch.zeros(self.num_bags-self.bag_idx, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device)],dim=0)
        for i in range(tokens.shape[0]-self.num_bags+1):
            if i == 0:
                time_sequence = tokens[i: i+self.num_bags].unsqueeze(0)
            else:
                time_sequence = torch.cat([time_sequence, tokens[i: i+self.num_bags].unsqueeze(0)], dim=0)
        return time_sequence

    def forward(self, x: torch.Tensor):
        res_0 = x[:, self.bag_idx]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        
        x = x[:, self.bag_idx]
        x = x + res_0
        res_1 = x
        x = self.generate_time_sequence(x)
        
        x = self.ln_pre_1(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_1(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post_1(x)

        x = x[:, self.bag_idx-1]#[anormal+normal,width]
        x = (x + res_1)
        x = self.fc(x)
        return x