"""
Volume loaders and the PyTorch ``Dataset`` for the verified MRI pipeline.

``load_vol`` / ``load_seg_mask`` handle NIfTI reading, intensity normalisation
and resampling.  ``GliomaDS`` performs tumour-centred cropping, augmentation,
and packages the multimodal sample (image + clinical/radiomic vector +
survival targets).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

from ..config import cfg


def load_vol(path, shape=None):
    """
    Load a NIfTI volume, clip+normalize to [0,1], resample to target shape.
    VERIFIED: handles all voxel types, empty masks, and shape mismatches.
    """
    shape = shape or cfg.TARGET_SHAPE
    img   = nib.load(path)
    arr   = img.get_fdata(dtype=np.float32)

    # Clip to [1st, 99th] percentile of foreground (non-zero) voxels
    fg = arr[arr > 0]
    if fg.size > 10:
        p1, p99 = np.percentile(fg, [1, 99])
        arr = np.clip(arr, p1, p99)
    # Normalize to [0, 1]
    d   = arr.max() - arr.min()
    arr = (arr - arr.min()) / (d + 1e-8)

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)          # [1,1,H,W,D]
    t = F.interpolate(t, size=shape, mode='trilinear', align_corners=False)
    return t.squeeze().numpy()                                    # [H,W,D]


def load_seg_mask(path, shape=None):
    """
    Load a segmentation mask, binarise (ANY label -> 1), resample with nearest.
    Returns float32 array in [0,1] - use threshold > 0.5 for binary.
    """
    shape = shape or cfg.TARGET_SHAPE
    img   = nib.load(path)
    arr   = img.get_fdata(dtype=np.float32)
    # Binarise before interpolation to avoid label-value artefacts
    arr   = (arr > 0).astype(np.float32)
    t     = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t     = F.interpolate(t, size=shape, mode='nearest')
    return t.squeeze().numpy()


class GliomaDS(Dataset):
    """
    Dataset that returns:
      image    : [2, H, W, D] float32  (FLAIR, T1ce - tumour-cropped)
      clinical : [CLIN_DIM] float32    (clinical + radiomic)
      os_days  : scalar
      log_os   : log1p(os_days)
      event    : 0/1
      bin_idx  : DeepHit time-bin index
      risk     : 0/1 (18-month binary)
    """
    def __init__(self, df, paths, feat_cols, time_bins, augment=False):
        self.df        = df.reset_index(drop=True)
        self.paths     = paths
        self.feat_cols = feat_cols
        self.time_bins = time_bins
        self.augment   = augment

    def _load_vol(self, idx):
        flair_p, t1c_p, seg_p = self.paths[idx]

        fl  = load_vol(flair_p)           # [H,W,D] in [0,1]
        t1c = load_vol(t1c_p)             # [H,W,D] in [0,1]
        seg = load_seg_mask(seg_p)        # [H,W,D] binary 0/1

        vol = np.stack([fl, t1c], axis=0)  # [2,H,W,D]

        # Tumour-centred crop with +/-12 voxel margin (preserves context)
        seg_bin = (seg > 0.5).astype(np.uint8)
        if seg_bin.sum() > 10:
            coords = np.argwhere(seg_bin)
            lo = np.maximum(coords.min(0) - 12, [0, 0, 0])
            hi = np.minimum(coords.max(0) + 12,
                            np.array(cfg.TARGET_SHAPE) - 1)
            # Crop - skip if crop is degenerate
            if np.all(hi > lo):
                vol = vol[:, lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]

        # Re-interpolate to target shape after crop
        t = torch.from_numpy(vol).unsqueeze(0)   # [1,2,H',W',D']
        t = F.interpolate(t, size=cfg.TARGET_SHAPE,
                          mode='trilinear', align_corners=False)
        return t.squeeze(0).numpy()               # [2,H,W,D]

    def _aug(self, vol):
        """Augmentation: flips, rotation, noise, gamma, cutout."""
        # Random flips
        for ax in (1, 2, 3):
            if np.random.rand() < 0.5:
                vol = np.flip(vol, ax).copy()
        # Random rotation (in-plane)
        if np.random.rand() < 0.3:
            vol = np.rot90(vol, np.random.randint(1, 4), axes=(1, 2)).copy()
        # Gaussian noise
        if np.random.rand() < 0.4:
            vol = np.clip(vol + np.random.randn(*vol.shape).astype(np.float32) * 0.03, 0, 1)
        # Gamma / contrast jitter
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.7, 1.4)
            vol   = np.power(np.clip(vol, 1e-8, 1.), gamma).astype(np.float32)
        # Per-channel intensity shift + scale
        if np.random.rand() < 0.3:
            for c in range(vol.shape[0]):
                shift = np.random.uniform(-0.05, 0.05)
                scale = np.random.uniform(0.90, 1.10)
                vol[c] = np.clip(vol[c] * scale + shift, 0, 1)
        # Cutout (random cuboid zeroing)
        if np.random.rand() < 0.25:
            _, H, W, D = vol.shape
            sh = np.random.randint(0, max(1, H // 4))
            sw = np.random.randint(0, max(1, W // 4))
            sd = np.random.randint(0, max(1, D // 4))
            eh = min(sh + H // 4, H)
            ew = min(sw + W // 4, W)
            ed = min(sd + D // 4, D)
            vol[:, sh:eh, sw:ew, sd:ed] = 0.
        return vol

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        vol    = self._load_vol(idx)
        if self.augment:
            vol = self._aug(vol)
        os_val = float(row['OS'])
        n_bins = len(self.time_bins)
        bin_i  = int(np.clip(
            np.searchsorted(self.time_bins, os_val, side='right') - 1,
            0, n_bins - 1))
        return dict(
            image    = torch.tensor(vol,   dtype=torch.float32),
            clinical = torch.tensor(
                row[self.feat_cols].values.astype(np.float32), dtype=torch.float32),
            os_days  = torch.tensor(os_val,              dtype=torch.float32),
            log_os   = torch.tensor(float(row['log_os']), dtype=torch.float32),
            event    = torch.tensor(float(row['1-dead 0-alive']), dtype=torch.float32),
            bin_idx  = torch.tensor(bin_i,               dtype=torch.long),
            risk     = torch.tensor(int(row['risk_label']), dtype=torch.long),
        )
