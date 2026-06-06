"""
Global configuration for MultiModalPreopSurvNet.

This module centralises every hyper-parameter, path and feature definition
used throughout the pipeline.  ``cfg`` and the feature-column lists are the
single source of truth for the whole package.
"""

import os
import torch

# --------------------------------------------------------------------------- #
# Torch backend setup                                                         #
# --------------------------------------------------------------------------- #
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Experiment configuration                                                    #
# --------------------------------------------------------------------------- #
class Cfg:

    CSV_PATH        = '/kaggle/input/datasets/usmansadiqcs/ucsf-pdgm-v3-dataset/UCSF-PDGM-metadata.csv'
    MRI_ROOT        = '/kaggle/input/datasets/usmansadiqcs/ucsf-pdgm-v3-dataset/UCSF-PDGM'
    OUT_DIR         = '/kaggle/working'
    MMAP_CKPT       = '/kaggle/working/mmps_mmap'
    CKPT_DIR        = '/kaggle/working/mmps_best_models'
    FIG_PATH        = '/kaggle/working/mmps_results.png'
    RESULTS_PATH    = '/kaggle/working/mmps_results.json'
    RAD_CACHE       = '/kaggle/working/mmps_radiomics_cache.pkl'

    TARGET_SHAPE = (96, 96, 64)
    N_CH         = 2           # FLAIR + T1ce
    HIGH_CUT     = 365 * 1.5   # 18-month threshold
    N_TIME_BINS  = 30
    N_FOLDS      = 5
    SEED         = 42

    N_EPOCHS  = 50
    MMAP_EP   = 15
    BATCH     = 8
    ACCUM     = 2
    LR        = 1e-4
    LR_ENC    = 1e-5
    LR_MIN    = 1e-6
    WD        = 5e-5
    PATIENCE  = 18
    WARMUP_EP = 5

    W_NLL     = 1.0
    W_RANK    = 0.5
    W_COX     = 0.5
    W_CALIB   = 0.8
    W_BRIER   = 0.3
    W_RISK    = 0.3
    W_RAD_L1  = 5e-5

    ISOTONIC_RECAL = True
    TTA_ENABLED    = True


cfg = Cfg()
os.makedirs(cfg.OUT_DIR,  exist_ok=True)
os.makedirs(cfg.CKPT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Feature definitions                                                         #
# --------------------------------------------------------------------------- #
CLINICAL_FEAT_COLS = [
    'sex_bin',       # demographic - known before surgery
    'age_norm',      # demographic - normalised per fold (see build_fold_scaler)
]
N_CLINICAL = len(CLINICAL_FEAT_COLS)

RAD_COLS = [
    # Shape - T1ce
    't1c_shape_VoxelVolume', 't1c_shape_Sphericity',
    't1c_shape_Elongation',  't1c_shape_Flatness',
    't1c_shape_SurfaceVolumeRatio', 't1c_shape_MaximumDiameter',
    # First-order - T1ce
    't1c_fo_Mean', 't1c_fo_Std', 't1c_fo_Skewness',
    't1c_fo_Kurtosis', 't1c_fo_Entropy', 't1c_fo_Energy',
    't1c_fo_10Percentile', 't1c_fo_90Percentile',
    # GLCM - T1ce
    't1c_glcm_Contrast', 't1c_glcm_Correlation',
    't1c_glcm_JointEnergy', 't1c_glcm_JointEntropy',
    't1c_glcm_Idm', 't1c_glcm_Imc1',
    # GLRLM - T1ce
    't1c_glrlm_ShortRunEmphasis', 't1c_glrlm_LongRunEmphasis',
    't1c_glrlm_GrayLevelNonUniformity',
    # GLSZM - T1ce
    't1c_glszm_SmallAreaEmphasis', 't1c_glszm_ZoneEntropy',
    # First-order - FLAIR
    'fl_fo_Mean', 'fl_fo_Std', 'fl_fo_Skewness',
    'fl_fo_Entropy', 'fl_fo_Energy',
    # GLCM - FLAIR
    'fl_glcm_Contrast', 'fl_glcm_JointEntropy', 'fl_glcm_Idm',
]
N_RAD     = len(RAD_COLS)            # 33
FEAT_COLS = CLINICAL_FEAT_COLS + RAD_COLS
CLIN_DIM  = len(FEAT_COLS)          # 2 + 33 = 35


def describe_features():
    """Print a short summary of the feature configuration."""
    print(f'Preoperative clinical features: {N_CLINICAL}  {CLINICAL_FEAT_COLS}')
    print(f'Radiomic features  : {N_RAD}')
    print(f'Total feature dim  : {CLIN_DIM}  '
          f'({N_CLINICAL} preop clinical + {N_RAD} radiomic)')
    print(f'RAD_COLS start idx : {N_CLINICAL}')
