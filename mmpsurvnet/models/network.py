"""
MultiModalPreopSurvNet architecture.

A 3-D ResNet image encoder with attention-based survival pooling, a clinical /
radiomic encoder, cross-modal attention fusion (CMAF), and three prediction
heads (discrete-time survival, Cox, 18-month risk).

  - cenc first layer maps CLIN_DIM features into a 256-dim hidden space.
  - cenc_rad_weights selects the radiomic columns [N_CLINICAL:] for L1 sparsity.
  - The CMAF gate is initialised at sigmoid(0.0) = 0.50 (balanced MRI/clinical).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CLIN_DIM, N_CLINICAL
from .layers import ConvBnSilu, ResBlock3D, SurvivalPool, CMAF


class MultiModalPreopSurvNet(nn.Module):
    """MRI (FLAIR + T1ce) + clinical + radiomic survival network."""
    def __init__(self, in_ch=2, clin_dim=None, base=32, drop=0.20, n_bins=30):
        super().__init__()
        clin_dim = clin_dim or CLIN_DIM
        D = base * 16   # 512

        # 3-D CNN image encoder
        self.enc0  = ConvBnSilu(in_ch, base,   k=7, s=2, p=3)
        self.enc1  = ResBlock3D(base,   base*2, stride=2, drop=drop * 0.5)
        self.enc2  = ResBlock3D(base*2, base*4, stride=2, drop=drop * 0.5)
        self.enc3  = ResBlock3D(base*4, base*8, stride=2, drop=drop * 0.5)
        self.enc4  = ResBlock3D(base*8, D,      stride=2, drop=drop * 0.5)
        self.pattn = SurvivalPool(D)

        # Image projection
        self.iproj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Dropout(drop))

        # Clinical + Radiomics encoder  (CLIN_DIM -> 256 -> 256)
        # The first Linear's weight[:, N_CLINICAL:] corresponds to radiomics - we
        # apply L1 regularisation to those weights in the training loop.
        self.cenc = nn.Sequential(
            nn.Linear(clin_dim, 256), nn.LayerNorm(256),
            nn.SiLU(), nn.Dropout(drop * 0.5),
            nn.Linear(256, 256),      nn.LayerNorm(256),
            nn.SiLU(), nn.Dropout(drop))

        # Cross-modal fusion
        self.fusion = CMAF(D, 256, heads=8, drop=drop)

        # Shared neck
        self.neck = nn.Sequential(
            nn.Linear(D,   256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(drop * 0.5),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU())

        # Prediction heads
        self.h_dh   = nn.Sequential(nn.Linear(128, 64), nn.SiLU(),
                                     nn.Dropout(0.15), nn.Linear(64, n_bins))
        self.h_cox  = nn.Sequential(nn.Linear(128, 64), nn.SiLU(),
                                     nn.Dropout(0.10), nn.Linear(64, 1))
        self.h_risk = nn.Sequential(nn.Linear(128, 64), nn.SiLU(),
                                     nn.Dropout(0.10), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def cenc_rad_weights(self):
        """Weights in the first cenc layer corresponding to radiomic features."""
        return self.cenc[0].weight[:, N_CLINICAL:]   # shape [256, N_RAD]

    def encode_img(self, x):
        x   = self.enc4(self.enc3(self.enc2(self.enc1(self.enc0(x)))))
        att = self.pattn(x)
        return self.iproj(att)

    def forward(self, image, clinical):
        img  = self.encode_img(image)
        clin = self.cenc(clinical)
        fuse = self.fusion(img, clin)
        feat = self.neck(fuse)
        sp   = F.softmax(self.h_dh(feat), dim=1)
        cif  = torch.cumsum(sp, dim=1)
        sf   = 1.0 - cif
        tk   = torch.arange(1, sf.size(1) + 1,
                            device=sf.device, dtype=torch.float32)
        risk = -(sf * tk).sum(1)
        return dict(surv_probs=sp, surv_fn=sf, risk_score=risk,
                    cox=self.h_cox(feat).squeeze(-1),
                    risk_logit=self.h_risk(feat).squeeze(-1))
