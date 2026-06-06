"""
MultiModalPreopSurvNet
======================

Preoperative multimodal survival modelling for glioblastoma on the UCSF-PDGM
cohort, combining MRI (FLAIR + T1ce), preoperative clinical variables and
radiomic features under a 5-fold cross-validation protocol.
"""

__version__ = '1.0.0'

from .config import (cfg, device, CLINICAL_FEAT_COLS, RAD_COLS, FEAT_COLS,
                     N_CLINICAL, N_RAD, CLIN_DIM)

__all__ = [
    'cfg', 'device',
    'CLINICAL_FEAT_COLS', 'RAD_COLS', 'FEAT_COLS',
    'N_CLINICAL', 'N_RAD', 'CLIN_DIM',
]
