import torch
from torch import nn

from models.basic_block import BasicLayer
from models.patch_embed import PatchEmbed
from models.patch_merging import PatchMerging


class SwinTransformer(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 n_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 n_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super(SwinTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # Stage4输出的channels，即embed_dim*8
        self.n_features = int(embed_dim * 2**(self.n_layers-1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size, in_c, embed_dim,
                                      norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(proj_dropout)

        # 根据深度递增dropout
        dpr = [x.item() for x in torch.linspace(0, dropout, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layers = BasicLayer(int(embed_dim*2**i), depths[i], n_heads[i],
                                window_size, mlp_ratio, qkv_bias,
                                proj_dropout, attn_dropout, 
                                dpr[sum(depths[:i]):sum(depths[:i+1])],
                                norm_layer, PatchMerging if i < self.n_layers-1 else None)
            self.layers.append(layers)

        self.norm = norm_layer(self.n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.n_features, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, h, w = layer(x, h, w)

        # (b,l,c)
        x = self.norm(x)
        # (b,l,c) -> (b,c,l) -> (b,c,1)
        x = self.avgpool(x.transpose(1, 2))
        # (b,c,1) -> (b,c)
        x = torch.flatten(x, 1)
        # (b,c) -> (b,n_classes)
        x = self.head(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)
