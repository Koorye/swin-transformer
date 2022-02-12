import torch
from torch import nn
from torch.nn import functional as F


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x, h, w):
        # (b,hw,c)
        b, l, c = x.shape
        # (b,hw,c) -> (b,h,w,c)
        x = x.view(b, h, w, c)

        # 如果h,w不是2的整数倍，需要padding
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        # (b,h/2,w/2,c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # (b,h/2,w/2,c)*4 -> (b,h/2,w/2,4c)
        x = torch.cat([x0, x1, x2, x3], -1)
        # (b,hw/4,4c)
        x = x.view(b, -1, 4*c)

        x = self.norm(x)
        # (b,hw/4,4c) -> (b,hw/4,2c)
        x = self.reduction(x)

        return x
