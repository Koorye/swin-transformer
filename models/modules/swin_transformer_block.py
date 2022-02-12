from numpy import pad
import torch
from torch import nn
from torch.nn import functional as F

from models.modules.mlp import MLP
from models.modules.window_attention import WindowAttention
from models.modules.window_utils import window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, n_heads,
                                    qkv_bias, attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim,
                       hid_features=dim*mlp_ratio,
                       dropout=proj_dropout)

    def forward(self, x, attn_mask):
        h, w = self.h, self.w
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # (n_windows*b,m,m,c)
        x_windows = window_partition(shifted_x, self.window_size)
        # (n_windows*b,m*m,c)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, c)

        attn_windows = self.attn(x_windows, attn_mask)

        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h*w, c)

        x = shortcut + self.dropout(x)
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x
