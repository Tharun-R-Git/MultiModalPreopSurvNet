"""
Reusable 3-D building blocks for MultiModalPreopSurvNet.

Contains the convolutional stem, squeeze-and-excitation / spatial attention,
residual block, attention-based survival pooling, and the Cross-Modal
Attention Fusion (CMAF) module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnSilu(nn.Sequential):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__(
            nn.Conv3d(ic, oc, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(min(32, oc), oc),
            nn.SiLU(inplace=True))


class SE3D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False), nn.SiLU(),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1, 1)


class SpatAttn3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(
            torch.cat([x.mean(1, keepdim=True),
                       x.max(1, keepdim=True)[0]], 1)))


class ResBlock3D(nn.Module):
    def __init__(self, ic, oc, stride=1, drop=0.10):
        super().__init__()
        self.c1   = ConvBnSilu(ic, oc, s=stride)
        self.c2   = ConvBnSilu(oc, oc)
        self.se   = SE3D(oc)
        self.sa   = SpatAttn3D()
        self.drop = nn.Dropout3d(drop)
        self.skip = (nn.Conv3d(ic, oc, 1, stride=stride, bias=False)
                     if (ic != oc or stride != 1) else nn.Identity())

    def forward(self, x):
        return F.silu(self.drop(self.sa(self.se(self.c2(self.c1(x))))) + self.skip(x))


class SurvivalPool(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(ch, ch // 4, 1), nn.SiLU(), nn.Conv3d(ch // 4, 1, 1))

    def forward(self, x):
        w = F.softmax(self.attn(x).view(x.size(0), 1, -1), -1)
        return (x.view(x.size(0), x.size(1), -1) * w).sum(-1)


class CMAF(nn.Module):
    """
    Cross-Modal Attention Fusion.
    The image gate is initialised at sigmoid(0.0) = 0.50 so the model starts with
    equal MRI/clinical weighting and learns the balance during training.
    """
    def __init__(self, img_dim, clin_dim, heads=8, drop=0.10):
        super().__init__()
        self.cp  = nn.Linear(clin_dim, img_dim)
        self.i2c = nn.MultiheadAttention(img_dim, heads, dropout=drop, batch_first=True)
        self.c2i = nn.MultiheadAttention(img_dim, heads, dropout=drop, batch_first=True)
        self.n1  = nn.LayerNorm(img_dim)
        self.n2  = nn.LayerNorm(img_dim)
        self.ff  = nn.Sequential(
            nn.Linear(img_dim, img_dim * 2), nn.SiLU(),
            nn.Dropout(drop), nn.Linear(img_dim * 2, img_dim))
        self.n3  = nn.LayerNorm(img_dim)
        # Init at 0.0 -> sigmoid(0.0) = 0.50  (balanced MRI / clinical start)
        self.img_gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, img, clin):
        c      = self.cp(clin).unsqueeze(1)
        i      = img.unsqueeze(1)
        i2, _  = self.i2c(i, c, c)
        i      = self.n1(i + i2)
        c2, _  = self.c2i(c, i, i)
        c      = self.n2(c + c2)
        gate   = torch.sigmoid(self.img_gate_logit)
        fused  = c.squeeze(1) + gate * i.squeeze(1)
        return self.n3(self.ff(fused) + fused)

    def get_gate(self):
        return float(torch.sigmoid(self.img_gate_logit).item())
