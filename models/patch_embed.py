from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 embed_dim=96,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 如果图片的H,W不是patch_size的整数倍，需要padding
        _, _, h, w = x.shape
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            x = F.pad(x, (0, self.patch_size - w % self.patch_size,
                          0, self.patch_size - h % self.patch_size,
                          0, 0))

        x = self.proj(x)
        _, _, h, w = x.shape

        # (b,c,h,w) -> (b,c,hw) -> (b,hw,c)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w
