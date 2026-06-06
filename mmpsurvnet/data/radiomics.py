"""
Radiomic feature extraction.

Uses PyRadiomics (van Griethuysen et al., 2017) when available, with a
NumPy-only fallback for shape / first-order descriptors otherwise.  Features
are extracted from the T1ce and FLAIR volumes within the tumour mask and
cached to disk.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm.auto import tqdm

from ..config import cfg, RAD_COLS

# --------------------------------------------------------------------------- #
# Optional PyRadiomics backend                                                #
# --------------------------------------------------------------------------- #
try:
    import radiomics  # noqa: F401
    from radiomics import featureextractor as _rad_extractor
    HAS_PYRAD = True
    logging.getLogger('radiomics').setLevel(logging.ERROR)
except ImportError:
    HAS_PYRAD = False
    print('WARNING: pyradiomics not found - will use fallback radiomics')


def _make_pyrad_extractor(modality='t1c'):
    params = {
        'setting': {
            'binWidth': 25,
            'resampledPixelSpacing': None,
            'interpolator': 'sitkBSpline',
            'normalizeScale': 1,
            'normalize': True,
            'removeOutliers': 3.0,
            'force2D': False,
            'label': 1,
        },
        'featureClass': {
            'firstorder':  [],
            'glcm':        [],
            'glrlm':       [],
            'glszm':       [],
        },
    }
    if modality == 't1c':
        params['featureClass']['shape'] = []

    ext = _rad_extractor.RadiomicsFeatureExtractor(params)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName('Original')
    return ext


def _extract_one_pyrad(extractor, img_path, seg_path, prefix):
    import SimpleITK as sitk
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
    seg = sitk.BinaryThreshold(seg, lowerThreshold=1,
                                upperThreshold=255, insideValue=1, outsideValue=0)
    try:
        result = extractor.execute(img, seg, label=1)
    except Exception:
        return {}
    feats = {}
    for k, v in result.items():
        if k.startswith('original_'):
            parts = k.split('_')
            if len(parts) >= 3:
                cls   = parts[1]
                name  = '_'.join(parts[2:])
                cls_map = {
                    'firstorder': 'fo', 'shape': 'shape',
                    'glcm': 'glcm', 'glrlm': 'glrlm', 'glszm': 'glszm'
                }
                cls_s = cls_map.get(cls, cls)
                key   = f'{prefix}_{cls_s}_{name}'
                try:
                    feats[key] = float(v)
                except Exception:
                    pass
    return feats


def _fallback_radiomics(t1c_path, seg_path):
    """NumPy-only shape / first-order descriptors used when PyRadiomics is absent."""
    from scipy.stats import skew as sc_skew, kurtosis as sc_kurt
    try:
        arr  = nib.load(t1c_path).get_fdata(dtype=np.float32)
        sarr = nib.load(seg_path).get_fdata()
        mask = (sarr > 0.5)
        if mask.sum() < 10:
            return {}
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0)
        dims   = (hi - lo + 1).astype(float)
        ds     = np.sort(dims)[::-1]
        vol    = float(mask.sum())
        surf   = float(2*(dims[0]*dims[1]+dims[1]*dims[2]+dims[0]*dims[2]))
        vox    = arr[mask].astype(np.float64)
        vn     = (vox - vox.min()) / (vox.max() - vox.min() + 1e-8)
        hist,_ = np.histogram(vn, bins=64, range=(0,1))
        pr     = hist / (hist.sum()+1e-8); pr = pr[pr > 0]
        feats  = {
            't1c_shape_VoxelVolume':        vol,
            't1c_shape_Sphericity':         float(ds[2]/(ds[1]+1e-8)),
            't1c_shape_Elongation':         float(ds[1]/(ds[0]+1e-8)),
            't1c_shape_Flatness':           float(ds[2]/(ds[0]+1e-8)),
            't1c_shape_SurfaceVolumeRatio': float(surf/(vol+1e-8)),
            't1c_shape_MaximumDiameter':    float(ds[0]),
            't1c_fo_Mean':                  float(np.mean(vn)),
            't1c_fo_Std':                   float(np.std(vn)),
            't1c_fo_Skewness':              float(sc_skew(vn)),
            't1c_fo_Kurtosis':              float(sc_kurt(vn)),
            't1c_fo_Entropy':               float(-np.sum(pr*np.log2(pr+1e-10))),
            't1c_fo_Energy':                float(np.sum(vn**2)/(len(vn)+1e-8)),
            't1c_fo_10Percentile':          float(np.percentile(vn, 10)),
            't1c_fo_90Percentile':          float(np.percentile(vn, 90)),
        }
        for c in RAD_COLS:
            feats.setdefault(c, 0.0)
        return feats
    except Exception:
        return {c: 0.0 for c in RAD_COLS}


def extract_radiomics(df_v, paths):
    if os.path.exists(cfg.RAD_CACHE):
        print('Radiomics cache found - loading')
        with open(cfg.RAD_CACHE, 'rb') as f:
            rd = pickle.load(f)
        print(f'  Shape: {rd.shape}')
        return rd

    print(f'Extracting radiomics for {len(df_v)} patients ...')
    if HAS_PYRAD:
        ext_t1c = _make_pyrad_extractor('t1c')
        ext_fl  = _make_pyrad_extractor('flair')
        print('  Using pyradiomics (official - van Griethuysen 2017)')
        print('  Settings: binWidth=25, normalize=True, removeOutliers=3sigma')
        print('  Classes: Shape(T1ce), FO+GLCM+GLRLM+GLSZM (T1ce+FLAIR)')
    else:
        print('  WARNING: pyradiomics not available - using numpy fallback')

    rows = []
    for i, (fl_p, t1c_p, seg_p) in enumerate(tqdm(paths, desc='Radiomics')):
        try:
            if HAS_PYRAD:
                feats_t1c = _extract_one_pyrad(ext_t1c, t1c_p, seg_p, 't1c')
                feats_fl  = _extract_one_pyrad(ext_fl,  fl_p,  seg_p, 'fl')
                feats     = {**feats_t1c, **feats_fl}
                row = {c: feats.get(c, 0.0) for c in RAD_COLS}
            else:
                row = _fallback_radiomics(t1c_p, seg_p)
                row = {c: row.get(c, 0.0) for c in RAD_COLS}
            rows.append(row)
        except Exception as e:
            print(f'  [{i}] error: {e}')
            rows.append({c: 0.0 for c in RAD_COLS})

        if (i+1) % 50 == 0:
            print(f'  Progress: {i+1}/{len(df_v)}')

    rd = pd.DataFrame(rows)[RAD_COLS]
    with open(cfg.RAD_CACHE, 'wb') as f:
        pickle.dump(rd, f)
    print(f'Radiomics extraction done -> {cfg.RAD_CACHE}')
    print(f'  Patients: {len(rd)} | Features: {len(rd.columns)}')
    return rd
