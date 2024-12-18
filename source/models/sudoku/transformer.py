import torch

import torch.nn as nn
import torch.nn.functional as F
import math
from source.layers.common_layers import Attention
from source.layers.common_fns import positionalencoding2d

from source.data.datasets.sudoku.sudoku import convert_onehot_to_int


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, hw=None, gta=True):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, weight="fc", randomgta=gta, hw=hw)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, src, T):
        # Repeat attention T times
        for _ in range(T):
            src2 = self.layernorm1(src)
            src2 = self.attn(src2, src2)
            src = src + src2

        src2 = self.layernorm2(src)
        src2 = self.mlp(src2)
        src = src + src2
        return src


class SudokuTransformer(nn.Module):
    def __init__(
        self,
        img_size=9,
        patch_size=1,
        num_classes=9,
        ch=64,
        blocks=6,
        heads=4,
        mlp_dim=128,
        R=1,
        T=1,
        gta=True,
    ):
        super().__init__()
        self.R = R
        self.T = T

        self.embedding = nn.Embedding(10, ch)
        self.pos_embed = nn.Parameter(
            positionalencoding2d(ch, img_size // patch_size, img_size // patch_size)
            .reshape(-1, img_size // patch_size, img_size // patch_size)
            .flatten(1, 2)
            .transpose(0, 1)
        )

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerBlock(
                    ch, heads, mlp_dim, 0.0, hw=[img_size, img_size], gta=gta
                )
                for _ in range(blocks)
            ]
        )

        self.out = torch.nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, num_classes))

    def forward(self, x, is_input):
        B = x.size(0)
        x = convert_onehot_to_int(x)
        x = self.embedding(x)
        x = x.view(B, -1, x.shape[-1])  # [B, H*W, C]
        is_input = is_input.view(B, -1)
        x = x + self.pos_embed[None]
        for block in self.transformer_encoder:
            x = block(x, self.T)
        x = self.out(x)
        x = x.view(-1, 9, 9, x.shape[-1])
        return x
