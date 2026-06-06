"""
Pooled out-of-fold (OOF) metrics with bootstrap 95% confidence intervals.

Aggregates every fold's OOF predictions, recomputes the survival /
classification metrics on the pooled cohort, runs a log-rank test on the
median-risk split, and bootstraps 95% CIs.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

from .config import cfg
from .metrics import (compute_ibs, compute_18month_brier,
                      compute_classification_metrics)


def compute_pooled_metrics(oof_records, pooled_ref_os, pooled_ref_ev,
                            n_bootstrap=500, seed=42):
    oof_df   = pd.DataFrame([{k: v for k, v in r.items() if k != 'surv_fn'}
                               for r in oof_records])
    surv_mat = np.array([r['surv_fn'] for r in oof_records])
    os_all   = oof_df['os'].values.astype(float)
    ev_all   = oof_df['event'].values.astype(float)
    risk_all = oof_df['risk_score'].values.astype(float)
    rp_all   = oof_df['risk_prob'].values.astype(float)
    rt_all   = oof_df['risk_true'].values.astype(int)

    harr  = float(concordance_index(os_all, -risk_all, ev_all))
    y_ref = Surv.from_arrays(pooled_ref_ev.astype(bool), pooled_ref_os)
    y_oof = Surv.from_arrays(ev_all.astype(bool), os_all)
    tau   = np.percentile(os_all, 90)
    try:
        cdt = float(concordance_index_ipcw(y_ref, y_oof, risk_all,
                                            tau=tau, tied_tol=1e-8)[0])
    except Exception as e:
        print(f'  IPCW Cdt fallback: {e}'); cdt = harr

    ibs  = compute_ibs(os_all, surv_mat, ev_all)
    b18  = compute_18month_brier(os_all, surv_mat, ev_all, cfg.HIGH_CUT)
    clf  = compute_classification_metrics(rt_all, rp_all)
    med  = np.median(risk_all)
    hi_m = risk_all >= med; lo_m = ~hi_m
    lr_r = logrank_test(os_all[hi_m], os_all[lo_m],
                         event_observed_A=ev_all[hi_m],
                         event_observed_B=ev_all[lo_m])

    rng = np.random.default_rng(seed); N = len(os_all)
    bs_harr, bs_cdt, bs_ibs, bs_auc, bs_b18 = [], [], [], [], []
    print(f'Bootstrap CI (n={n_bootstrap})...')
    for _ in tqdm(range(n_bootstrap), desc='Bootstrap'):
        idx_b  = rng.choice(N, N, replace=True)
        os_b   = os_all[idx_b];   ev_b   = ev_all[idx_b]
        risk_b = risk_all[idx_b]; surv_b = surv_mat[idx_b]
        rp_b   = rp_all[idx_b];   rt_b   = rt_all[idx_b]
        try:    bs_harr.append(float(concordance_index(os_b, -risk_b, ev_b)))
        except: bs_harr.append(harr)
        try:
            ri  = rng.choice(len(pooled_ref_os), len(pooled_ref_os), replace=True)
            yrb = Surv.from_arrays(pooled_ref_ev[ri].astype(bool), pooled_ref_os[ri])
            yb  = Surv.from_arrays(ev_b.astype(bool), os_b)
            tb  = np.percentile(os_b, 90)
            bs_cdt.append(float(concordance_index_ipcw(yrb, yb, risk_b, tau=tb)[0]))
        except: bs_cdt.append(cdt)
        try:    bs_ibs.append(compute_ibs(os_b, surv_b, ev_b))
        except: bs_ibs.append(ibs)
        try:    bs_b18.append(compute_18month_brier(os_b, surv_b, ev_b, cfg.HIGH_CUT))
        except: bs_b18.append(b18)
        try:    bs_auc.append(roc_auc_score(rt_b, rp_b))
        except: bs_auc.append(clf['auc'])

    def ci(vals, fb):
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 10: return (fb, fb)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return dict(
        harrell=harr, cdt=cdt, ibs=ibs, brier_18m=b18,
        auc=clf['auc'], sensitivity=clf['sensitivity'],
        specificity=clf['specificity'], f1=clf['f1'],
        logrank_p=float(lr_r.p_value),
        ci_harrell=ci(bs_harr, harr), ci_cdt=ci(bs_cdt, cdt),
        ci_ibs=ci(bs_ibs, ibs), ci_brier_18m=ci(bs_b18, b18),
        ci_auc=ci(bs_auc, clf['auc']),
        os_all=os_all, ev_all=ev_all, risk_all=risk_all,
        rp_all=rp_all, rt_all=rt_all, surv_mat=surv_mat,
    )
