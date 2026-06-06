"""
5-fold stratified cross-validation driver.

For each fold the following are derived from the training fold only:
  - the feature StandardScaler,
  - the discrete survival time bins,
  - the masked-autoencoder (MMAP) encoder weights,
  - the isotonic recalibrators,
after which the model is trained and out-of-fold (OOF) predictions collected.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import cfg, device, N_CLINICAL, N_RAD, CLIN_DIM
from .data.scaling import build_fold_scaler
from .models.mmap import run_mmap_for_fold
from .train import train_one_fold


def run_5fold_cv(df_valid, all_paths):
    print('\n' + '='*72)
    print('MultiModalPreopSurvNet - 5-Fold CV | UCSF-PDGM (Preoperative)')
    print(f'Features: {N_CLINICAL} preop-clinical + {N_RAD} radiomic = {CLIN_DIM} total')
    print(f'Dataset: {len(df_valid)} patients  |  Device: {device}')
    print('='*72)

    skf   = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    strat = df_valid['strat_label'].values
    idx   = np.arange(len(df_valid))

    fold_results  = []
    oof_records   = []
    train_ref_os  = []
    train_ref_ev  = []
    all_gate_hist = []

    for fold_idx, (tr_i, te_i) in enumerate(skf.split(idx, strat)):
        print(f'\n{"="*72}')
        print(f'  FOLD {fold_idx+1}/{cfg.N_FOLDS}  train={len(tr_i)}  test={len(te_i)}')
        print(f'{"="*72}')

        df_tr_raw = df_valid.iloc[tr_i].copy().reset_index(drop=True)
        df_te_raw = df_valid.iloc[te_i].copy().reset_index(drop=True)
        paths_tr  = [all_paths[i] for i in tr_i]
        paths_te  = [all_paths[i] for i in te_i]

        # Feature scaler fit on the training fold
        df_tr_sc, df_te_sc, *_ = build_fold_scaler(df_tr_raw, df_te_raw)

        # Survival time bins from training-fold OS
        os_train  = df_tr_raw['OS'].values.astype(np.float32)
        time_bins = np.unique(
            np.percentile(os_train,
                          np.linspace(5, 95, cfg.N_TIME_BINS)).astype(np.float32))
        print(f'  Time bins: N={len(time_bins)} '
              f'[{time_bins[0]:.0f}-{time_bins[-1]:.0f}] days')

        # MMAP pretraining on the training-fold images
        mmap_ckpt = run_mmap_for_fold(fold_idx, df_tr_sc, paths_tr)

        oof_res, history, gate_hist, tr_os, tr_ev, _ = train_one_fold(
            fold_idx, df_tr_sc, df_te_sc, paths_tr, paths_te,
            time_bins, mmap_ckpt, verbose=True)

        train_ref_os.append(tr_os)
        train_ref_ev.append(tr_ev)
        all_gate_hist.append(gate_hist)

        best_ep = next((h['epoch'] for h in reversed(history) if h['is_best']),
                       cfg.N_EPOCHS)
        fold_results.append({
            'fold':        fold_idx + 1,
            'n_train':     len(tr_i),
            'n_test':      len(te_i),
            'harrell':     oof_res['harrell'],
            'cdt':         oof_res['cdt'],
            'ibs':         oof_res['ibs'],
            'brier_18m':   oof_res.get('brier_18m', float('nan')),
            'auc':         oof_res['clf']['auc'],
            'sensitivity': oof_res['clf']['sensitivity'],
            'specificity': oof_res['clf']['specificity'],
            'f1':          oof_res['clf']['f1'],
            'best_epoch':  best_ep,
            'final_gate':  gate_hist[-1] if gate_hist else float('nan'),
        })
        for j in range(len(oof_res['os'])):
            oof_records.append({
                'fold':       fold_idx + 1,
                'orig_idx':   int(te_i[j]),
                'os':         oof_res['os'][j],
                'event':      oof_res['event'][j],
                'risk_score': oof_res['risk_score'][j],
                'risk_prob':  oof_res['risk_prob'][j],
                'risk_true':  oof_res['risk_true'][j],
                'surv_fn':    oof_res['surv'][j].tolist(),
            })
        print(f'  Fold {fold_idx+1}: Cdt={oof_res["cdt"]:.4f}  '
              f'IBS={oof_res["ibs"]:.4f}  '
              f'AUC={oof_res["clf"]["auc"]:.4f}  '
              f'Gate={gate_hist[-1] if gate_hist else float("nan"):.3f}')

    pooled_ref_os = np.concatenate(train_ref_os)
    pooled_ref_ev = np.concatenate(train_ref_ev)
    return (fold_results, oof_records,
            pooled_ref_os, pooled_ref_ev, all_gate_hist)
