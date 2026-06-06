"""
Metadata loading and patient discovery for UCSF-PDGM.

Reads the clinical CSV, derives preoperative labels / stratification targets,
and pairs each surviving patient with verified FLAIR / T1ce / segmentation
NIfTI volumes on disk.
"""

import os
import re
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib

from ..config import cfg


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['OS', '1-dead 0-alive']).reset_index(drop=True)

    df['sex_bin']      = (df['Sex'] == 'M').astype(float)
    df['age_raw']      = df['Age at MRI'].astype(float)
    df['age_norm']     = 0.0

    df['risk_label'] = (df['OS'] < cfg.HIGH_CUT).astype(int)
    df['log_os']     = np.log1p(df['OS'].astype(float))

    os_bins = pd.cut(
        df['OS'],
        bins=[0, 180, 365, 548, np.inf],   # 6m / 12m / 18m thresholds
        labels=['s0', 's1', 's2', 's3'],
        right=True,
    ).astype(str)
    df['strat_label'] = (
        df['1-dead 0-alive'].astype(int).astype(str) + '_' + os_bins
    )

    print(f'Metadata: {len(df)} patients | '
          f'Median OS: {df["OS"].median():.0f} days | '
          f'Events: {int(df["1-dead 0-alive"].sum())}/{len(df)} | '
          f'High-risk (< {cfg.HIGH_CUT:.0f}d): {int(df["risk_label"].sum())}')
    print('Preoperative clinical features: sex, age')
    return df


def _verify_nifti(path):
    try:
        img   = nib.load(path)
        shp   = img.shape
        zooms = img.header.get_zooms()[:3]
        n_vox = int(np.prod(shp))
        if n_vox == 0:
            return False, f'EMPTY shape={shp}'
        if os.path.getsize(path) < 1024:
            return False, f'FILE_TOO_SMALL ({os.path.getsize(path)} bytes)'
        return True, f'shape={shp} zoom={tuple(round(z,2) for z in zooms)}'
    except Exception as e:
        return False, f'READ_ERROR: {e}'


def discover_patients(df, mri_root, verbose=False):
    avail = {}
    if os.path.isdir(mri_root):
        for name in os.listdir(mri_root):
            fp = os.path.join(mri_root, name)
            if os.path.isdir(fp) and '_FU' not in name:
                avail[name] = fp

    id_col = next((c for c in ['ID', 'Patient ID', 'Subject ID']
                   if c in df.columns), None)

    valid_rows, valid_paths = [], []
    skipped = defaultdict(int)

    for _, row in df.iterrows():
        pid   = str(row[id_col]) if id_col else str(row.name)
        m     = re.search(r'(\d+)', pid)
        pid_n = m.group(1) if m else pid
        pdir  = next((fp for fn, fp in avail.items() if pid_n in fn), None)
        if pdir is None:
            skipped['no_directory'] += 1
            continue

        niis = [f for f in glob.glob(os.path.join(pdir, '**', '*.nii*'),
                                      recursive=True)
                if os.path.isfile(f) and os.path.getsize(f) > 0]

        flair = [f for f in niis if 'FLAIR' in f and 'bias' in f.lower()]
        if not flair:
            flair = [f for f in niis if 'FLAIR' in f and 'seg' not in f.lower()]
        if not flair:
            flair = [f for f in niis if 'FLAIR' in f]

        t1c_kw = ['T1c', 'T1CE', 'T1Gd', 't1gd', 'gad', 'ce', 'contrast']
        t1c    = [f for f in niis
                  if any(x in f for x in t1c_kw) and 'bias' in f.lower()
                  and 'seg' not in f.lower()]
        if not t1c:
            t1c = [f for f in niis
                   if any(x in f for x in t1c_kw) and 'seg' not in f.lower()]
        if not t1c:
            t1c = [f for f in niis if any(x in f for x in t1c_kw)]

        seg_kw = ['tumor_seg', 'segmentation', '_seg', '_mask', 'tumor_mask']
        seg    = [f for f in niis if any(x in f.lower() for x in seg_kw)]

        if not (flair and t1c and seg):
            skipped['missing_modality'] += 1
            if verbose:
                print(f'  Skip {pid}: FLAIR={len(flair)} T1c={len(t1c)} seg={len(seg)}')
            continue

        flair_ok, flair_info = _verify_nifti(flair[0])
        t1c_ok,   t1c_info   = _verify_nifti(t1c[0])
        seg_ok,   seg_info   = _verify_nifti(seg[0])

        if not (flair_ok and t1c_ok and seg_ok):
            skipped['corrupt_file'] += 1
            if verbose:
                print(f'  Skip {pid}: FLAIR={flair_info} T1c={t1c_info} seg={seg_info}')
            continue

        valid_rows.append(row)
        valid_paths.append((flair[0], t1c[0], seg[0]))

    df_v = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f'Valid (FLAIR + T1ce + seg): {len(df_v)}/{len(df)}')
    print(f'  Skipped: {dict(skipped)}')
    return df_v, valid_paths
