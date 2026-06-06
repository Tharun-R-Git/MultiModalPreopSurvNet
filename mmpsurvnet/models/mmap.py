"""
Masked Autoencoder Pretraining (MMAP).

Self-supervised reconstruction pretraining of the 3-D image encoder.  A random
~25% voxel mask is applied and the decoder reconstructs the masked regions.
Pretraining is run per-fold on the training images of that fold.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ..config import cfg, device, FEAT_COLS, CLIN_DIM
from ..data.dataset import GliomaDS
from .network import MultiModalPreopSurvNet


class MaskedMAP(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        for n in ['enc0', 'enc1', 'enc2', 'enc3', 'enc4']:
            setattr(self, n, getattr(encoder, n))
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(True),
            nn.ConvTranspose3d(128, cfg.N_CH, 4, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        B, C, H, W, D = x.shape
        ms   = (torch.rand(B, 1, H//8, W//8, D//8, device=x.device) > 0.75).float()
        mask = F.interpolate(ms, size=(H, W, D), mode='nearest')
        xm   = x * (1 - mask)
        feat = self.enc4(self.enc3(self.enc2(self.enc1(self.enc0(xm)))))
        feat = feat.mean(dim=[2, 3, 4], keepdim=True)
        rec  = F.interpolate(self.dec(feat), size=(H, W, D),
                             mode='trilinear', align_corners=False)
        return rec, mask


def run_mmap_for_fold(fold_idx, df_tr_sc, paths_tr):
    fold_ckpt = f'{cfg.MMAP_CKPT}_fold{fold_idx+1}.pth'
    if os.path.exists(fold_ckpt):
        print(f'  [MMAP F{fold_idx+1}] checkpoint found - skipping')
        return torch.load(fold_ckpt, map_location=device)

    print(f'  [MMAP F{fold_idx+1}] pretraining on {len(df_tr_sc)} images '
          f'| {cfg.MMAP_EP} epochs (train fold only)')

    base  = MultiModalPreopSurvNet(clin_dim=CLIN_DIM, n_bins=cfg.N_TIME_BINS).to(device)
    mmap  = MaskedMAP(base).to(device)
    opt_m = torch.optim.AdamW(mmap.parameters(), lr=5e-5, weight_decay=1e-5)
    sch_m = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_m, T_max=cfg.MMAP_EP, eta_min=1e-6)
    scl_m = GradScaler()

    dummy_bins = np.linspace(0, 1000, cfg.N_TIME_BINS).astype(np.float32)
    ds_m  = GliomaDS(df_tr_sc, paths_tr, FEAT_COLS, dummy_bins, augment=True)
    ld_m  = DataLoader(ds_m, batch_size=4, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True, persistent_workers=True)

    best_m = float('inf')
    for ep in range(1, cfg.MMAP_EP + 1):
        mmap.train(); tot = 0.
        for batch in ld_m:
            img = batch['image'].to(device)
            opt_m.zero_grad()
            with autocast():
                rec, msk = mmap(img)
                loss = (F.mse_loss(rec, img, reduction='none') * msk).mean()
            scl_m.scale(loss).backward()
            scl_m.step(opt_m); scl_m.update()
            tot += loss.item()
        avg = tot / len(ld_m)
        sch_m.step()
        best_m = min(best_m, avg)
        print(f'  [MMAP F{fold_idx+1}] ep {ep:2d}/{cfg.MMAP_EP} '
              f'loss={avg:.5f} best={best_m:.5f}')

    ckpt = {n: getattr(mmap, n).state_dict()
            for n in ['enc0', 'enc1', 'enc2', 'enc3', 'enc4']}
    torch.save(ckpt, fold_ckpt)
    del base, mmap, opt_m, sch_m, scl_m
    torch.cuda.empty_cache()
    print(f'  [MMAP F{fold_idx+1}] saved -> {fold_ckpt}')
    return ckpt


def load_mmap_weights(model, ckpt):
    for n in ['enc0', 'enc1', 'enc2', 'enc3', 'enc4']:
        try:
            getattr(model, n).load_state_dict(ckpt[n])
        except Exception as e:
            print(f'  MMAP load warn [{n}]: {e}')
    return model
