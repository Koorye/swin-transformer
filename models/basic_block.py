import numpy as np
import torch
from torch import nn

from models.modules.swin_transformer_block import SwinTransformerBlock
from models.modules.window_utils import window_partition


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 n_heads,
                 window_size,
                 mlp_ratio=4,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        # 窗口向右和向下的移动数为窗口宽度除以2向下取整
        self.shift_size = window_size // 2

        # 按照每个Stage的深度堆叠若干Block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, n_heads, window_size,
                                 0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio, qkv_bias, proj_dropout, attn_dropout,
                                 dropout[i] if isinstance(
                                     dropout, list) else dropout,
                                 norm_layer)
            for i in range(depth)])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, h, w):
        attn_mask = self.create_mask(x, h, w)
        for blk in self.blocks:
            blk.h, blk.w = h, w
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, h, w)
            # 如果是奇数，相当于做padding后除以2
            # 如果是偶数，相当于直接除以2
            h, w = (h+1) // 2, (w+1) // 2

        return x, h, w

    def create_mask(self, x, h, w):
        # 保证hp,wp是window_size的整数倍
        hp = int(np.ceil(h/self.window_size)) * self.window_size
        wp = int(np.ceil(w/self.window_size)) * self.window_size

        # (1,hp,wp,1)
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)

        # 将feature map分割成9个区域
        # 例如，对于9x9图片
        # 若window_size=3, shift_size=3//2=1
        # 得到slices为([0,-3],[-3,-1],[-1,])
        # 于是h从0至-4(不到-3)，w从0至-4
        # 即(0,0)(-4,-4)围成的矩形为第1个区域
        # h从0至-4，w从-3至-2
        # 即(0,-3)(-4,-2)围成的矩形为第2个区域...
        # h\w 0 1 2 3 4 5 6 7 8
        # --+--------------------
        # 0 | 0 0 0 0 0 0 1 1 2
        # 1 | 0 0 0 0 0 0 1 1 2
        # 2 | 0 0 0 0 0 0 1 1 2
        # 3 | 0 0 0 0 0 0 1 1 2
        # 4 | 0 0 0 0 0 0 1 1 2
        # 5 | 0 0 0 0 0 0 1 1 2
        # 6 | 3 3 3 3 3 3 4 4 5
        # 7 | 3 3 3 3 3 3 4 4 5
        # 8 | 6 6 6 6 6 6 7 7 8
        # 这样在每个窗口内，相同数字的区域都是连续的
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # (1,hp,wp,1) -> (n_windows,m,m,1) m表示window_size
        mask_windows = window_partition(img_mask, self.window_size)
        # (n_windows,m,m,1) -> (n_windows,m*m)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)

        # (n_windows,1,m*m) - (n_windows,m*m,1)
        # -> (n_windows,m*m,m*m)
        # 以window
        # [[4 4 5]
        #  [4 4 5]
        #  [7 7 8]]
        # 为例
        # 展平后为 [4,4,5,4,4,5,7,7,8]
        # [[4,4,5,4,4,5,7,7,8]] - [[4]
        #                          [4]
        #                          [5]
        #                          [4]
        #                          [4]
        #                          [5]
        #                          [7]
        #                          [7]
        #                          [8]]
        # -> [[0,0,-,0,0,-,-,-,-]
        #     [0,0,-,0,0,-,-,-,-]
        #     [...]]
        # 于是有同样数字的区域为0，不同数字的区域为非0
        # attn_mask[1,3]即为窗口的第3个元素(1,0)和第1个元素(0,1)是否相同
        # 若相同，则值为0，否则为非0
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将非0的元素设为-100
        attn_mask = (attn_mask
                     .masked_fill(attn_mask != 0, float(-100.))
                     .masked_fill(attn_mask == 0, float(0.)))

        return attn_mask

