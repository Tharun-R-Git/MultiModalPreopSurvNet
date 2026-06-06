"""
Per-fold training loop.

Builds the model (with MMAP-initialised encoder), optimises the composite
survival objective with differential encoder/head learning rates, warmup +
cosine schedule, mixed precision and gradient accumulation, selects the best
epoch by the composite score, fits the per-bin isotonic calibrators on the
training fold, and produces the final (TTA + calibrated) out-of-fold
predictions.
"""

import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .config import cfg, device, FEAT_COLS, CLIN_DIM, N_CLINICAL, N_RAD
from .data.dataset import GliomaDS
from .models.network import MultiModalPreopSurvNet
from .models.mmap import load_mmap_weights
from .losses import (SurvCalibLoss, dh_fn, cox_fn, rank_fn, bce_fn, brier_fn)
from .metrics import (evaluate_fold, fit_isotonic_calibration,
                      compute_ibs, compute_18month_brier)


def get_lr_factor(epoch):
    if epoch <= cfg.WARMUP_EP:
        return epoch / cfg.WARMUP_EP
    t = (epoch - cfg.WARMUP_EP) / (cfg.N_EPOCHS - cfg.WARMUP_EP)
    return cfg.LR_MIN / cfg.LR + (1 - cfg.LR_MIN / cfg.LR) * 0.5 * (1 + np.cos(np.pi * t))


def train_one_fold(fold_idx, df_tr, df_te, paths_tr, paths_te,
                   time_bins, mmap_ckpt, verbose=True):
    n_bins   = len(time_bins)
    tr_os    = df_tr['OS'].values.astype(float)
    tr_ev    = df_tr['1-dead 0-alive'].values.astype(float)
    time_bins_tensor = torch.tensor(time_bins, dtype=torch.float32, device=device)

    tr_ds       = GliomaDS(df_tr, paths_tr, FEAT_COLS, time_bins, augment=True)
    te_ds       = GliomaDS(df_te, paths_te, FEAT_COLS, time_bins, augment=False)
    tr_ds_noaug = GliomaDS(df_tr, paths_tr, FEAT_COLS, time_bins, augment=False)

    kw = dict(num_workers=2, pin_memory=True, persistent_workers=True)
    tr_ld       = DataLoader(tr_ds,       batch_size=cfg.BATCH, shuffle=True,
                              drop_last=True, **kw)
    te_ld       = DataLoader(te_ds,       batch_size=cfg.BATCH, shuffle=False, **kw)
    tr_ld_noaug = DataLoader(tr_ds_noaug, batch_size=cfg.BATCH, shuffle=False, **kw)

    model = MultiModalPreopSurvNet(clin_dim=CLIN_DIM, n_bins=n_bins).to(device)
    model = load_mmap_weights(model, mmap_ckpt)

    calib_fn = SurvCalibLoss(K=n_bins, n_eval_pts=cfg.N_TIME_BINS).to(device)

    enc_names = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4']
    enc_p = [p for n, p in model.named_parameters()
             if any(n.startswith(x) for x in enc_names)]
    rst_p = [p for n, p in model.named_parameters()
             if not any(n.startswith(x) for x in enc_names)]
    optimizer = torch.optim.AdamW(
        [{'params': enc_p, 'lr': cfg.LR_ENC},
         {'params': rst_p, 'lr': cfg.LR}],
        weight_decay=cfg.WD)
    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_factor)
    scaler_g   = GradScaler()

    best_composite = -float('inf')
    best_weights   = None
    patience_cnt   = 0
    history        = []
    gate_history   = []
    fold_ckpt_path = os.path.join(cfg.CKPT_DIR, f'best_fold{fold_idx+1}.pth')

    print(f'  {"-"*65}')
    print(f'  Fold {fold_idx+1}  |  {cfg.N_EPOCHS} epochs  |  patience {cfg.PATIENCE}')
    print(f'  Features: {N_CLINICAL} preop-clinical + {N_RAD} radiomic = {CLIN_DIM} total')
    print(f'  Loss: NLL + Rank + Cox + Calib + Brier + Risk + RadL1')
    print(f'  {"-"*65}')

    for epoch in range(1, cfg.N_EPOCHS + 1):
        model.train(); tot_loss = 0.; optimizer.zero_grad()
        for bi, batch in enumerate(tr_ld):
            img    = batch['image'].to(device)
            clin   = batch['clinical'].to(device)
            os_d   = batch['os_days'].to(device)
            ev     = batch['event'].to(device)
            bn     = batch['bin_idx'].to(device)
            risk_t = batch['risk'].to(device)
            log_os = batch['log_os'].to(device)

            with autocast():
                out = model(img, clin)
                # Main survival losses
                L_surv = (cfg.W_NLL   * dh_fn(out['surv_probs'], bn, ev)
                        + cfg.W_RANK  * rank_fn(out['risk_score'], os_d, ev)
                        + cfg.W_COX   * cox_fn(out['cox'], log_os, ev)
                        + cfg.W_CALIB * calib_fn(out['surv_fn'], os_d, ev)
                        + cfg.W_BRIER * brier_fn(out['surv_fn'], os_d,
                                                   ev, time_bins_tensor)
                        + cfg.W_RISK  * bce_fn(out['risk_logit'], risk_t.float()))
                # Radiomics L1 (feature selection on cenc first-layer rad weights)
                L_rad_l1 = cfg.W_RAD_L1 * model.cenc_rad_weights.abs().mean()
                L = (L_surv + L_rad_l1) / cfg.ACCUM

            scaler_g.scale(L).backward()
            if (bi + 1) % cfg.ACCUM == 0:
                scaler_g.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_g.step(optimizer); scaler_g.update()
                optimizer.zero_grad()
            tot_loss += L.item() * cfg.ACCUM

        if len(tr_ld) % cfg.ACCUM != 0:
            scaler_g.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_g.step(optimizer); scaler_g.update()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = tot_loss / len(tr_ld)

        # Validation (without TTA - faster; TTA used for final evaluation only)
        val_res   = evaluate_fold(model, te_ld, tr_os, tr_ev, use_tta=False)
        val_cdt   = val_res['cdt']
        val_ibs   = val_res['ibs'] if not np.isnan(val_res['ibs']) else 0.20
        val_auc   = val_res['clf']['auc']
        # Composite selection score: Cdt - 0.5*IBS + 0.1*AUC
        composite = val_cdt - 0.5 * val_ibs + 0.1 * val_auc
        gate_val  = model.fusion.get_gate()
        gate_history.append(gate_val)

        is_best = composite > best_composite
        if is_best:
            best_composite = composite
            best_weights   = copy.deepcopy(model.state_dict())
            patience_cnt   = 0
            torch.save({'state_dict': best_weights, 'epoch': epoch,
                        'cdt': val_cdt, 'ibs': val_ibs,
                        'gate': gate_val, 'fold': fold_idx + 1},
                       fold_ckpt_path)
        else:
            patience_cnt += 1

        if verbose:
            mark = ' *' if is_best else ''
            print(f'  F{fold_idx+1} ep{epoch:3d}/{cfg.N_EPOCHS} '
                  f'loss={avg_loss:.4f} Cdt={val_cdt:.4f} '
                  f'IBS={val_ibs:.4f} AUC={val_auc:.4f} '
                  f'gate={gate_val:.3f} pat={patience_cnt}{mark}')

        history.append({'epoch': epoch, 'loss': avg_loss, 'cdt': val_cdt,
                        'ibs': val_res['ibs'], 'auc': val_auc,
                        'composite': composite, 'is_best': is_best,
                        'gate': gate_val})

        if patience_cnt >= cfg.PATIENCE:
            print(f'  Early stop epoch {epoch} | best composite={best_composite:.4f}')
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f'  Best model -> {fold_ckpt_path}  |  Final gate: {model.fusion.get_gate():.4f}')

    # Isotonic recalibration: calibrators fit on the training fold
    iso_cals = None
    if cfg.ISOTONIC_RECAL:
        print(f'  [Calib] Fitting calibrators on train fold...')
        train_res = evaluate_fold(model, tr_ld_noaug, tr_os, tr_ev,
                                   isotonic_cals=None, use_tta=False)
        iso_cals  = fit_isotonic_calibration(
            train_res['surv'], train_res['os'], train_res['event'])
        print(f'  [Calib] Fitted on {len(train_res["os"])} train samples.')

    # Final test evaluation: with calibration + TTA
    oof_res = evaluate_fold(model, te_ld, tr_os, tr_ev,
                            isotonic_cals=iso_cals,
                            use_tta=cfg.TTA_ENABLED)
    if iso_cals is not None:
        try:
            oof_res['ibs'] = compute_ibs(oof_res['os'], oof_res['surv'], oof_res['event'])
        except Exception: pass
        try:
            oof_res['brier_18m'] = compute_18month_brier(
                oof_res['os'], oof_res['surv'], oof_res['event'], cfg.HIGH_CUT)
        except Exception: pass
        print(f'  Post-cal: IBS={oof_res["ibs"]:.4f}  '
              f'Brier@18m={oof_res.get("brier_18m", float("nan")):.4f}  '
              f'AUC={oof_res["clf"]["auc"]:.4f}')

    del model, scaler_g, optimizer, scheduler
    torch.cuda.empty_cache()
    return oof_res, history, gate_history, tr_os, tr_ev, iso_cals
