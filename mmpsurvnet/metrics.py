"""
Evaluation metrics, isotonic recalibration, and the fold evaluator.

  - compute_cdt                 : Harrell C and IPCW time-dependent C-index
  - compute_ibs                 : integrated Brier score (IPCW)
  - compute_18month_brier       : Brier score at the 18-month horizon
  - compute_classification_metrics : AUC / sensitivity / specificity / F1
  - fit/apply_isotonic_calibration : per-bin isotonic recalibration
  - evaluate_fold               : full forward pass + optional TTA + metrics
"""

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

from .config import cfg, device


def compute_cdt(os_tr, ev_tr, os_te, ev_te, risk_te):
    y_tr  = Surv.from_arrays(ev_tr.astype(bool), os_tr)
    y_te  = Surv.from_arrays(ev_te.astype(bool), os_te)
    tau   = np.percentile(os_te, 90)
    cdt   = concordance_index_ipcw(y_tr, y_te, risk_te, tau=tau, tied_tol=1e-8)[0]
    harr  = concordance_index(os_te, -risk_te, ev_te)
    return float(harr), float(cdt)


def compute_ibs(os_days, surv_mat, events):
    N, K    = surv_mat.shape
    t_pts   = np.linspace(os_days.min(), os_days.max(), K)
    km_cens = KaplanMeierFitter()
    km_cens.fit(os_days, event_observed=(1 - events))
    eval_t  = np.linspace(np.percentile(os_days, 10),
                           np.percentile(os_days, 90), 100)
    brier   = []
    for t in eval_t:
        G_t = max(float(km_cens.predict(t)), 1e-4)
        bs  = 0.
        for i in range(N):
            S_t = float(np.clip(np.interp(t, t_pts, surv_mat[i]), 0, 1))
            if os_days[i] <= t and events[i] == 1:
                G_Ti = max(float(km_cens.predict(os_days[i])), 1e-4)
                bs  += (0. - S_t) ** 2 / G_Ti
            elif os_days[i] > t:
                bs  += (1. - S_t) ** 2 / G_t
        brier.append(bs / N)
    if len(brier) < 2: return float('nan')
    return float(np.trapz(brier, eval_t) / (eval_t[-1] - eval_t[0]))


def compute_18month_brier(os_days, surv_mat, events, high_cut=365 * 1.5):
    N, K  = surv_mat.shape
    t_pts = np.linspace(os_days.min(), os_days.max(), K)
    t     = high_cut
    km_c  = KaplanMeierFitter()
    km_c.fit(os_days, event_observed=(1 - events))
    G_t   = max(float(km_c.predict(t)), 1e-4)
    bs    = 0.
    for i in range(N):
        S_t = float(np.clip(np.interp(t, t_pts, surv_mat[i]), 0, 1))
        if os_days[i] <= t and events[i] == 1:
            G_Ti = max(float(km_c.predict(os_days[i])), 1e-4)
            bs  += (0. - S_t) ** 2 / G_Ti
        elif os_days[i] > t:
            bs  += (1. - S_t) ** 2 / G_t
    return float(bs / N)


def compute_classification_metrics(risk_true, risk_prob):
    try:
        auc           = roc_auc_score(risk_true, risk_prob)
        fpr, tpr, thr = roc_curve(risk_true, risk_prob)
        j = tpr - fpr; oi = np.argmax(j); ot = thr[oi]
        pred = (risk_prob >= ot).astype(int)
        tn, fp, fn, tp = confusion_matrix(risk_true, pred).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1   = f1_score(risk_true, pred, zero_division=0)
        return dict(auc=float(auc), sensitivity=float(sens),
                    specificity=float(spec), f1=float(f1),
                    opt_threshold=float(ot))
    except Exception:
        return dict(auc=0.5, sensitivity=0., specificity=0.,
                    f1=0., opt_threshold=0.5)


def fit_isotonic_calibration(surv_mat_train, os_days_train, events_train):
    """Fit per-bin isotonic calibrators on TRAIN fold."""
    from sklearn.isotonic import IsotonicRegression
    N, K  = surv_mat_train.shape
    t_pts = np.linspace(os_days_train.min(), os_days_train.max(), K)
    km_c  = KaplanMeierFitter()
    km_c.fit(os_days_train, event_observed=(1 - events_train))
    calibrators = []
    for k in range(K):
        t_k  = float(t_pts[k])
        G_tk = max(float(km_c.predict(t_k)), 1e-4)
        y_t  = np.zeros(N, dtype=np.float32)
        w    = np.ones(N,  dtype=np.float32)
        for i in range(N):
            if os_days_train[i] > t_k:
                y_t[i] = 1.; w[i] = 1. / G_tk
            elif events_train[i] == 1:
                y_t[i] = 0.
                G_Ti   = max(float(km_c.predict(os_days_train[i])), 1e-4)
                w[i]   = 1. / G_Ti
            else:
                w[i] = 0.
        x_pred = surv_mat_train[:, k].astype(np.float32)
        if w.sum() < 5:
            calibrators.append(None); continue
        ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
        try:
            ir.fit(x_pred, y_t, sample_weight=w)
            calibrators.append(ir)
        except Exception:
            calibrators.append(None)
    return calibrators


def apply_isotonic_calibration(surv_mat, calibrators):
    sc = surv_mat.copy()
    for k, cal in enumerate(calibrators):
        if cal is not None:
            sc[:, k] = cal.predict(surv_mat[:, k].astype(np.float32))
    for i in range(sc.shape[0]):
        for k in range(1, sc.shape[1]):
            sc[i, k] = min(sc[i, k], sc[i, k - 1])
    return sc.clip(0, 1)


@torch.no_grad()
def evaluate_fold(model, loader, tr_os, tr_ev,
                  isotonic_cals=None, use_tta=False):
    """
    Evaluate a model on a loader.
    If use_tta=True, averages predictions from:
      - original
      - horizontal flip (axis=2)
      - vertical flip (axis=3)
    """
    model.eval()
    surv_l, risk_l, os_l, ev_l, rp_l, rt_l = [], [], [], [], [], []

    for batch in loader:
        img   = batch['image'].to(device)
        clin  = batch['clinical'].to(device)

        if use_tta:
            # Original
            out0 = model(img, clin)
            sf0  = out0['surv_fn'].cpu().float()
            rs0  = out0['risk_score'].cpu().float()
            rl0  = out0['risk_logit'].cpu().float()
            # H-flip
            out1 = model(img.flip(2), clin)
            sf1  = out1['surv_fn'].cpu().float()
            rs1  = out1['risk_score'].cpu().float()
            rl1  = out1['risk_logit'].cpu().float()
            # W-flip
            out2 = model(img.flip(3), clin)
            sf2  = out2['surv_fn'].cpu().float()
            rs2  = out2['risk_score'].cpu().float()
            rl2  = out2['risk_logit'].cpu().float()
            # Ensemble (mean)
            sf_  = ((sf0 + sf1 + sf2) / 3.0)
            rs_  = ((rs0 + rs1 + rs2) / 3.0)
            rl_  = ((rl0 + rl1 + rl2) / 3.0)
        else:
            out  = model(img, clin)
            sf_  = out['surv_fn'].cpu().float()
            rs_  = out['risk_score'].cpu().float()
            rl_  = out['risk_logit'].cpu().float()

        surv_l.extend(sf_.tolist())
        risk_l.extend(rs_.tolist())
        rp_l.extend(torch.sigmoid(rl_).tolist())
        os_l.extend(batch['os_days'].tolist())
        ev_l.extend(batch['event'].tolist())
        rt_l.extend(batch['risk'].tolist())

    os_a   = np.array(os_l);   ev_a   = np.array(ev_l)
    risk_a = np.array(risk_l); surv_a = np.array(surv_l)

    if isotonic_cals is not None:
        surv_a = apply_isotonic_calibration(surv_a, isotonic_cals)

    try:    harr, cdt = compute_cdt(tr_os, tr_ev, os_a, ev_a, risk_a)
    except: harr = cdt = 0.5
    try:    ibs = compute_ibs(os_a, surv_a, ev_a)
    except: ibs = float('nan')
    try:    b18 = compute_18month_brier(os_a, surv_a, ev_a, cfg.HIGH_CUT)
    except: b18 = float('nan')
    clf = compute_classification_metrics(np.array(rt_l), np.array(rp_l))

    return dict(harrell=harr, cdt=cdt, ibs=ibs, brier_18m=b18,
                clf=clf, surv=surv_a, risk_score=risk_a,
                os=os_a, event=ev_a,
                risk_prob=np.array(rp_l), risk_true=np.array(rt_l))
