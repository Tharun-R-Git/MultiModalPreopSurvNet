"""
Reporting utilities: console results table, the 8-panel publication figure,
and the results JSON export.
"""

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from sklearn.metrics import roc_curve
from lifelines import KaplanMeierFitter

from .config import cfg, N_CLINICAL, N_RAD, CLIN_DIM, CLINICAL_FEAT_COLS, RAD_COLS


def print_results_table(fold_results, pooled):
    print('\n' + '='*90)
    print('MultiModalPreopSurvNet - 5-Fold CV | UCSF-PDGM | MRI + Preop-Clinical + Radiomics')
    print(f'Features: {N_CLINICAL} preop-clinical + {N_RAD} radiomic = {CLIN_DIM} total')
    print('='*90)
    hdr = (f'{"Fold":<5} {"N-te":<6} {"HarrC":<8} {"IPCW-C":<8} {"IBS":<8}'
           f'{"Brier18m":<10} {"AUC":<8} {"Gate":<7} {"BestEp":<7}')
    print(hdr); print('-'*90)
    for r in fold_results:
        b18s = f'{r.get("brier_18m", float("nan")):.4f}'
        gts  = f'{r.get("final_gate", float("nan")):.3f}'
        print(f'{r["fold"]:<5} {r["n_test"]:<6} {r["harrell"]:<8.4f}'
              f'{r["cdt"]:<8.4f} {r["ibs"]:<8.4f} {b18s:<10}'
              f'{r["auc"]:<8.4f} {gts:<7} {r["best_epoch"]:<7}')
    print('-'*90)
    for key, label in [('harrell', 'HarrC'), ('cdt', 'IPCW-C'),
                        ('ibs', 'IBS'), ('auc', 'AUC')]:
        vals = [r[key] for r in fold_results]
        print(f'{label}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  '
              f'range=[{min(vals):.4f}, {max(vals):.4f}]')
    print('='*90)
    print(f'\nPOOLED OOF (n={len(pooled["os_all"])}) - Bootstrap 95% CI:')
    print(f'  Harrell C  : {pooled["harrell"]:.4f}  '
          f'[{pooled["ci_harrell"][0]:.4f}, {pooled["ci_harrell"][1]:.4f}]')
    print(f'  IPCW Cdt   : {pooled["cdt"]:.4f}  '
          f'[{pooled["ci_cdt"][0]:.4f}, {pooled["ci_cdt"][1]:.4f}]')
    print(f'  IBS        : {pooled["ibs"]:.4f}  '
          f'[{pooled["ci_ibs"][0]:.4f}, {pooled["ci_ibs"][1]:.4f}]')
    b18 = pooled.get('brier_18m', float('nan'))
    print(f'  Brier@18mo : {b18:.4f}  '
          f'[{pooled["ci_brier_18m"][0]:.4f}, {pooled["ci_brier_18m"][1]:.4f}]')
    print(f'  AUC @18mo  : {pooled["auc"]:.4f}  '
          f'[{pooled["ci_auc"][0]:.4f}, {pooled["ci_auc"][1]:.4f}]')
    print(f'  Sensitivity: {pooled["sensitivity"]:.4f}  '
          f'Specificity: {pooled["specificity"]:.4f}  '
          f'F1: {pooled["f1"]:.4f}')
    print(f'  Log-rank p : {pooled["logrank_p"]:.2e}')
    print('='*90)


def make_publication_figure(fold_results, pooled, all_gate_hist, fig_path):
    PAL  = ['#1565C0', '#C62828', '#2E7D32', '#E65100', '#6A1B9A']
    GREY = '#607D8B'
    fig  = plt.figure(figsize=(22, 14))
    gs   = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.42)
    fig.suptitle(
        f'MultiModalPreopSurvNet  |  Glioblastoma Survival  |  UCSF-PDGM  n={len(pooled["os_all"])}\n'
        f'5-Fold CV  |  MRI + Preop-Clinical ({N_CLINICAL}) + Radiomic ({N_RAD}) features',
        fontsize=10, fontweight='bold')

    os_all    = pooled['os_all'];   ev_all   = pooled['ev_all']
    risk_all  = pooled['risk_all']; rp_all   = pooled['rp_all']
    rt_all    = pooled['rt_all'];   surv_mat = pooled['surv_mat']
    fold_cdts = [r['cdt']  for r in fold_results]
    fold_ibss = [r['ibs']  for r in fold_results]
    fold_b18  = [r.get('brier_18m', float('nan')) for r in fold_results]
    fold_aucs = [r['auc']  for r in fold_results]

    # Panel A - Kaplan-Meier
    ax = fig.add_subplot(gs[0, 0])
    med = np.median(risk_all); hi_m = risk_all >= med; lo_m = ~hi_m
    for mask, lbl, col in [(hi_m, 'High risk', PAL[1]), (lo_m, 'Low risk', PAL[0])]:
        km = KaplanMeierFitter()
        km.fit(os_all[mask] / 30.44, event_observed=ev_all[mask], label=lbl)
        km.plot_survival_function(ax=ax, ci_show=True, color=col, linewidth=2)
    ax.set_xlabel('Time (months)', fontsize=9); ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_title(f'A  Kaplan-Meier (pooled OOF)\nLog-rank p={pooled["logrank_p"]:.2e}',
                 fontsize=9, fontweight='bold')
    ax.set_xlim(0, 30); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B - ROC
    ax = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(rt_all, rp_all)
    ax.plot(fpr, tpr, color=PAL[0], lw=2.5,
            label=f'AUC={pooled["auc"]:.3f} [{pooled["ci_auc"][0]:.3f},{pooled["ci_auc"][1]:.3f}]')
    ax.fill_between(fpr, tpr, alpha=0.12, color=PAL[0])
    ax.plot([0, 1], [0, 1], color=GREY, ls='--', lw=1)
    ax.set_xlabel('1 - Specificity', fontsize=9); ax.set_ylabel('Sensitivity', fontsize=9)
    ax.set_title('B  ROC Curve (18-month risk)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # Panel C - Calibration
    ax    = fig.add_subplot(gs[0, 2])
    K     = surv_mat.shape[1]
    t_pts = np.linspace(os_all.min(), os_all.max(), K)
    km_c  = KaplanMeierFitter(); km_c.fit(os_all, event_observed=ev_all)
    mpred = surv_mat.mean(axis=0)
    km_s  = np.array([float(km_c.predict(t)) for t in t_pts])
    t_mo  = t_pts / 30.44
    ax.plot(t_mo, km_s,  color=PAL[1], lw=2.5, label='KM observed')
    ax.plot(t_mo, mpred, color=PAL[0], lw=2, ls='--', label='Mean predicted')
    ax.fill_between(t_mo, km_s, mpred, where=(mpred >= km_s), alpha=0.15,
                    color='orange', label='Over-predict')
    ax.fill_between(t_mo, km_s, mpred, where=(mpred < km_s),  alpha=0.15,
                    color='steelblue', label='Under-predict')
    b18str = f'{pooled.get("brier_18m", float("nan")):.4f}'
    ax.set_xlabel('Time (months)', fontsize=9); ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_title(f'C  Calibration  |  IBS={pooled["ibs"]:.4f}  Brier@18m={b18str}',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right'); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # Panel D - Risk distribution
    ax = fig.add_subplot(gs[0, 3])
    ev1 = risk_all[ev_all == 1]; ev0 = risk_all[ev_all == 0]
    ax.hist(ev1, bins=30, alpha=0.65, color=PAL[1], density=True, label='Event')
    ax.hist(ev0, bins=30, alpha=0.65, color=PAL[0], density=True, label='Censored')
    ax.axvline(np.median(risk_all), color='black', lw=2, ls='--', label='Median')
    ax.set_xlabel('Risk score', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.set_title('D  Risk Score Distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel E - Per-fold metrics
    ax = fig.add_subplot(gs[1, 0])
    x  = np.arange(len(fold_results))
    w  = 0.2
    ax.bar(x - 1.5*w, fold_cdts, w, color=PAL[0], alpha=0.85, label='IPCW-Cdt')
    ax.bar(x - 0.5*w, fold_ibss, w, color=PAL[2], alpha=0.85, label='IBS')
    ax.bar(x + 0.5*w, [v if not np.isnan(v) else 0. for v in fold_b18],
           w, color=PAL[3], alpha=0.85, label='Brier@18m')
    ax.bar(x + 1.5*w, fold_aucs, w, color=PAL[4], alpha=0.85, label='AUC')
    ax.set_xticks(x); ax.set_xticklabels([f'F{r["fold"]}' for r in fold_results])
    ax.set_xlabel('Fold', fontsize=9); ax.set_ylabel('Metric', fontsize=9)
    ax.set_title(f'E  Per-fold Metrics\nCdt={np.mean(fold_cdts):.4f}+/-{np.std(fold_cdts):.4f}  '
                 f'AUC={np.mean(fold_aucs):.4f}+/-{np.std(fold_aucs):.4f}',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # Panel F - Cdt vs IBS per fold
    ax = fig.add_subplot(gs[1, 1])
    colors_f = PAL[:cfg.N_FOLDS]
    for fi, (cdts_f, ibss_f) in enumerate(zip(fold_cdts, fold_ibss)):
        ax.bar(fi + 1 - 0.15, cdts_f, 0.28, color=colors_f[fi], alpha=0.8)
        ax.bar(fi + 1 + 0.15, ibss_f, 0.28, color=colors_f[fi], alpha=0.4, hatch='//')
    handles = [Patch(facecolor='grey', alpha=0.8, label='Cdt (solid)'),
               Patch(facecolor='grey', alpha=0.4, hatch='//', label='IBS (hatch)')]
    ax.legend(handles=handles, fontsize=8); ax.set_xticks(range(1, 6))
    ax.set_xlabel('Fold', fontsize=9); ax.set_ylabel('Value', fontsize=9)
    ax.set_title('F  Cdt vs IBS per Fold', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel G - Image gate evolution
    ax = fig.add_subplot(gs[1, 2])
    for fi, gh in enumerate(all_gate_hist):
        if gh:
            ax.plot(range(1, len(gh) + 1), gh, color=PAL[fi % 5], lw=1.5,
                    alpha=0.8, label=f'Fold {fi+1}')
    ax.axhline(0.5, color=GREY, ls='--', lw=1.5, alpha=0.7, label='Balanced (0.5)')
    ax.set_xlabel('Epoch', fontsize=9); ax.set_ylabel('Image gate value', fontsize=9)
    ax.set_title('G  CMAF Image Gate per Epoch\n(>0.5 = MRI-dominant, <0.5 = clinical-dominant)',
                 fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel H - Individual survival curves
    ax     = fig.add_subplot(gs[1, 3])
    t_pts2 = np.linspace(os_all.min(), os_all.max(), surv_mat.shape[1]) / 30.44
    order  = np.argsort(risk_all); low10 = order[:10]; high10 = order[-10:]
    for i, idx_s in enumerate(low10):
        ax.plot(t_pts2, surv_mat[idx_s], color=PAL[0], lw=0.8, alpha=0.6,
                label='Low risk' if i == 0 else '_')
    for i, idx_s in enumerate(high10):
        ax.plot(t_pts2, surv_mat[idx_s], color=PAL[1], lw=0.8, alpha=0.6,
                label='High risk' if i == 0 else '_')
    ax.set_xlabel('Time (months)', fontsize=9); ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_title('H  Individual Survival Curves\n(10 lowest / 10 highest risk)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure saved -> {fig_path}')
    plt.close(fig)


def save_results_json(fold_results, pooled, path):
    cdts  = [r['cdt']  for r in fold_results]
    ibss  = [r['ibs']  for r in fold_results]
    aucs  = [r['auc']  for r in fold_results]
    b18s  = [r.get('brier_18m', float('nan')) for r in fold_results]
    gates = [r.get('final_gate', float('nan')) for r in fold_results]

    results = dict(
        model         = 'MultiModalPreopSurvNet',
        description   = (
            f'Preoperative multimodal survival model: MRI (FLAIR + T1ce) + '
            f'{N_CLINICAL} preoperative clinical + {N_RAD} radiomic features. '
            'Per-fold feature scaling, survival time bins, MMAP pretraining and '
            'isotonic recalibration are all derived from the training fold.'
        ),
        dataset   = 'UCSF-PDGM',
        features  = dict(
            n_preop_clinical = N_CLINICAL,
            n_radiomic  = N_RAD,
            n_total     = CLIN_DIM,
            preop_clinical   = CLINICAL_FEAT_COLS,
            radiomic    = RAD_COLS,
        ),
        protocol  = dict(
            cv             = f'{cfg.N_FOLDS}-fold stratified',
            n_epochs       = cfg.N_EPOCHS,
            mmap_epochs    = cfg.MMAP_EP,
            patience       = cfg.PATIENCE,
            tta_enabled    = cfg.TTA_ENABLED,
            isotonic_recal = cfg.ISOTONIC_RECAL,
        ),
        per_fold     = fold_results,
        mean_cdt     = float(np.mean(cdts)),
        std_cdt      = float(np.std(cdts)),
        mean_ibs     = float(np.mean(ibss)),
        std_ibs      = float(np.std(ibss)),
        mean_auc     = float(np.mean(aucs)),
        std_auc      = float(np.std(aucs)),
        mean_brier18 = float(np.nanmean(b18s)),
        mean_gate    = float(np.nanmean(gates)),
        pooled_oof   = dict(
            n_patients   = int(len(pooled['os_all'])),
            harrell_c    = float(pooled['harrell']),
            harrell_ci95 = list(pooled['ci_harrell']),
            cdt          = float(pooled['cdt']),
            cdt_ci95     = list(pooled['ci_cdt']),
            ibs          = float(pooled['ibs']),
            ibs_ci95     = list(pooled['ci_ibs']),
            brier_18m    = float(pooled.get('brier_18m', float('nan'))),
            brier_18m_ci = list(pooled['ci_brier_18m']),
            auc_18m      = float(pooled['auc']),
            auc_ci95     = list(pooled['ci_auc']),
            sensitivity  = float(pooled['sensitivity']),
            specificity  = float(pooled['specificity']),
            f1_score     = float(pooled['f1']),
            logrank_p    = float(pooled['logrank_p']),
        ),
    )
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved -> {path}')
