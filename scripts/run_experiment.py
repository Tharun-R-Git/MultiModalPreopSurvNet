"""
End-to-end experiment runner for MultiModalPreopSurvNet.

Pipeline:

    1. Load metadata and discover verified MRI / segmentation volumes.
    2. Extract (or load cached) radiomic features and merge them in.
    3. Run 5-fold cross-validation.
    4. Compute pooled out-of-fold metrics with bootstrap 95% CIs.
    5. Print the results table, save the 8-panel publication figure and the
       results JSON.

Usage
-----
    python -m scripts.run_experiment

Paths and hyper-parameters are defined in ``mmpsurvnet/config.py`` (``Cfg``).
"""

import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

import torch

from mmpsurvnet.config import (cfg, device, describe_features,
                               CLIN_DIM, RAD_COLS, N_RAD, N_CLINICAL)
from mmpsurvnet.data import load_metadata, discover_patients, extract_radiomics
from mmpsurvnet.models import MultiModalPreopSurvNet
from mmpsurvnet.cross_validation import run_5fold_cv
from mmpsurvnet.evaluate import compute_pooled_metrics
from mmpsurvnet.reporting import (print_results_table, make_publication_figure,
                                  save_results_json)


def build_dataset():
    """Load metadata, discover volumes, and merge radiomic features."""
    describe_features()
    print(f'Device: {device}  |  PyTorch {torch.__version__}')

    df_all              = load_metadata(cfg.CSV_PATH)
    df_valid, all_paths = discover_patients(df_all, cfg.MRI_ROOT, verbose=False)

    rad_df = extract_radiomics(df_valid, all_paths)
    for col in RAD_COLS:
        df_valid[col] = rad_df[col].values

    print(f'\nRadiomic features merged (N={N_RAD})')
    print(f'Any NaN in radiomics: {df_valid[RAD_COLS].isna().any().any()}')
    print(f'Total feature dim: {CLIN_DIM}  '
          f'({N_CLINICAL} preop clinical + {N_RAD} radiomic)')
    return df_valid, all_paths


def sanity_check_model():
    """Instantiate the model once to report its parameter count."""
    m      = MultiModalPreopSurvNet(n_bins=cfg.N_TIME_BINS, clin_dim=CLIN_DIM).to(device)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'MultiModalPreopSurvNet: {params:,} trainable parameters')
    print(f'Clinical+Radiomic encoder: {CLIN_DIM} -> 256 -> 256  '
          f'({N_CLINICAL} preop-clinical + {N_RAD} radiomic)')
    print(f'Gate init: {m.fusion.get_gate():.4f}  (balanced MRI/clinical start)')
    del m
    torch.cuda.empty_cache()


def main():
    # ---- Data ------------------------------------------------------------- #
    df_valid, all_paths = build_dataset()
    sanity_check_model()

    # ---- 5-fold cross-validation ----------------------------------------- #
    (fold_results, oof_records,
     pooled_ref_os, pooled_ref_ev,
     all_gate_hist) = run_5fold_cv(df_valid, all_paths)

    # ---- Pooled OOF metrics ---------------------------------------------- #
    print('\nComputing pooled OOF metrics...')
    pooled = compute_pooled_metrics(oof_records, pooled_ref_os, pooled_ref_ev,
                                    n_bootstrap=500)

    # ---- Reporting -------------------------------------------------------- #
    print_results_table(fold_results, pooled)
    make_publication_figure(fold_results, pooled, all_gate_hist, cfg.FIG_PATH)
    save_results_json(fold_results, pooled, cfg.RESULTS_PATH)

    print('\n' + '='*72)
    print('MultiModalPreopSurvNet COMPLETE')
    print(f'  Figure   -> {cfg.FIG_PATH}')
    print(f'  Results  -> {cfg.RESULTS_PATH}')
    print(f'  Pooled Cdt  = {pooled["cdt"]:.4f}  [{pooled["ci_cdt"][0]:.4f}, {pooled["ci_cdt"][1]:.4f}]')
    print(f'  Pooled IBS  = {pooled["ibs"]:.4f}  [{pooled["ci_ibs"][0]:.4f}, {pooled["ci_ibs"][1]:.4f}]')
    print(f'  Pooled AUC  = {pooled["auc"]:.4f}  [{pooled["ci_auc"][0]:.4f}, {pooled["ci_auc"][1]:.4f}]')
    print(f'  Log-rank p  = {pooled["logrank_p"]:.2e}')
    print('='*72)


if __name__ == '__main__':
    main()
