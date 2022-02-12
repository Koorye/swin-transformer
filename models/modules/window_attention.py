import torch
from torch import nn


class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 n_heads,
                 qkv_bias=True,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -.5

        # ((2m-1)*(2m-1),n_heads)
        # 相对位置参数表长为(2m-1)*(2m-1)
        # 行索引和列索引各有2m-1种可能，故其排列组合有(2m-1)*(2m-1)种可能
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1) * (2*window_size - 1),
                        n_heads))

        # 构建窗口的绝对位置索引
        # 以window_size=2为例
        # coord_h = coord_w = [0,1]
        # meshgrid([0,1], [0,1])
        # -> [[0,0], [[0,1]
        #     [1,1]], [0,1]]
        # -> [[0,0,1,1],
        #     [0,1,0,1]]
        # (m,)
        coord_h = torch.arange(self.window_size)
        coord_w = torch.arange(self.window_size)
        # (m,)*2 -> (m,m)*2 -> (2,m,m)
        coords = torch.stack(torch.meshgrid([coord_h, coord_w]))
        # (2,m*m)
        coord_flatten = torch.flatten(coords, 1)

        # 构建窗口的相对位置索引
        # (2,m*m,1) - (2,1,m*m) -> (2,m*m,m*m)
        # 以coord_flatten为
        # [[0,0,1,1]
        #  [0,1,0,1]]为例
        # 对于第一个元素[0,0,1,1]
        # [[0],[0],[1],[1]] - [[0,0,1,1]]
        # -> [[0,0,0,0] - [[0,0,1,1] = [[0,0,-1,-1]
        #     [0,0,0,0]    [0,0,1,1]    [0,0,-1,-1]
        #     [1,1,1,1]    [0,0,1,1]    [1,1, 0, 0]
        #     [1,1,1,1]]   [0,0,1,1]]   [1,1, 0, 0]]
        # 相当于每个元素的h减去每个元素的h
        # 例如，第一行[0,0,0,0] - [0,0,1,1] -> [0,0,-1,-1]
        # 即为元素(0,0)相对(0,0)(0,1)(1,0)(1,1)为列(h)方向的差
        # 第二个元素即为每个元素的w减去每个元素的w
        # 于是得到窗口内每个元素相对每个元素高和宽的差
        # 例如relative_coords[0,1,2]
        # 即为窗口的第1个像素(0,1)和第2个像素(1,0)在列(h)方向的差
        relative_coords = coord_flatten[:, :, None] - coord_flatten[:, None, :]
        # (m*m,m*m,2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # 论文中提到的，将二维相对位置索引转为一维的过程
        # 1. 行列都加上m-1
        # 2. 行乘以2m-1
        # 3. 行列相加
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # (m*m,m*m,2) -> (m*m,m*m)
        relative_pos_idx = relative_coords.sum(-1)
        self.register_buffer('relative_pos_idx', relative_pos_idx)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        # (b*n_windows,m*m,total_embed_dim)
        # -> (b*n_windows,m*m,3*total_embed_dim)
        # -> (b*n_windows,m*m,3,n_heads,embed_dim_per_head)
        # -> (3,b*n_windows,n_heads,m*m,embed_dim_per_head)
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4))
        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        # (b*n_windows,n_heads,m*m,m*m)
        attn = (q @ k.transpose(-2, -1))

        # (m*m*m*m,n_heads)
        # -> (m*m,m*m,n_heads)
        # -> (n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m) + (1,n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m)
        relative_pos_bias = (self.relative_position_bias_table[self.relative_pos_idx.view(-1)]
                             .view(self.window_size*self.window_size, self.window_size*self.window_size, -1))
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_pos_bias.unsqueeze(0)

        if mask is not None:
            # mask: (n_windows,m*m,m*m)
            nw = mask.shape[0]
            # (b*n_windows,n_heads,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            # + (1,n_windows,1,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            attn = (attn.view(b//nw, nw, self.n_heads, n, n)
                    + mask.unsqueeze(1).unsqueeze(0))
            # (b,n_windows,n_heads,m*m,m*m)
            # -> (b*n_windows,n_heads,m*m,m*m)
            attn = attn.view(-1, self.n_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        # -> (b*n_windows,m*m,n_heads,embed_dim_per_head)
        # -> (b*n_windows,m*m,total_embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
