"""
Survival and classification loss functions.

  - DeepHitNLL    : discrete-time negative log-likelihood
  - CoxLoss       : Cox partial likelihood (Breslow)
  - PairRankLoss  : differentiable concordance (pairwise ranking)
  - SurvCalibLoss : IPCW-weighted calibration against Kaplan-Meier
  - IBSBrierLoss  : integrated / 18-month Brier score (IPCW)
  - FocalBCE      : focal binary cross-entropy for 18-month risk head

The module also instantiates the shared singleton loss objects
(``dh_fn``, ``cox_fn``, ``rank_fn``, ``bce_fn``, ``brier_fn``) on ``device``.
``SurvCalibLoss`` is constructed per-fold inside the training loop (its bin
count depends on the fold).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg, device


class DeepHitNLL(nn.Module):
    def forward(self, surv_probs, bin_idx, event):
        eps   = 1e-7
        B, K  = surv_probs.shape
        p_evt = surv_probs[torch.arange(B), bin_idx].clamp(eps, 1 - eps)
        cif   = torch.cumsum(surv_probs, dim=1)
        s_ti  = (1. - cif[torch.arange(B), bin_idx]).clamp(eps, 1 - eps)
        return (-(event * torch.log(p_evt) + (1 - event) * torch.log(s_ti))).mean()


class CoxLoss(nn.Module):
    def forward(self, log_hr, time, event):
        log_hr, time, event = log_hr.float(), time.float(), event.float()
        order  = torch.argsort(time, descending=True)
        log_hr = log_hr[order] - log_hr.max().detach()
        event  = event[order]
        return (-(log_hr - torch.logcumsumexp(log_hr, 0)) * event).sum() \
               / (event.sum() + 1e-8)


class PairRankLoss(nn.Module):
    def __init__(self, margin=0.1, sigma=0.1, max_p=12):
        super().__init__()
        self.margin = margin; self.sigma = sigma; self.max_p = max_p

    def forward(self, risk, time, event):
        risk, time, event = risk.float(), time.float(), event.float()
        total = torch.tensor(0., device=risk.device); cnt = 0
        for i in torch.where(event > 0.5)[0]:
            jj = torch.where(time > time[i])[0]
            if not len(jj): continue
            if len(jj) > self.max_p:
                jj = jj[torch.randperm(len(jj), device=jj.device)[:self.max_p]]
            total += torch.sigmoid(
                (risk[jj] - risk[i] + self.margin) / self.sigma).sum()
            cnt += len(jj)
        return total / (cnt + 1e-8)


class SurvCalibLoss(nn.Module):
    def __init__(self, K, n_eval_pts=30):
        super().__init__()
        self.K = K; self.n_eval_pts = n_eval_pts

    @staticmethod
    def _km_and_censor(times_np, events_np, eval_times_np):
        order = np.argsort(times_np)
        t_ord = times_np[order]; e_ord = events_np[order]
        n     = len(t_ord)
        km_out = np.ones(len(eval_times_np), dtype=np.float32)
        g_out  = np.ones(len(eval_times_np), dtype=np.float32)
        km_val = 1.0; prev_t = -np.inf; i = 0
        for ei, t_ev in enumerate(eval_times_np):
            while i < n and t_ord[i] <= t_ev:
                if t_ord[i] != prev_t:
                    ar = n - i
                    d  = int(np.sum(e_ord[i:][t_ord[i:] == t_ord[i]]))
                    if ar > 0 and d > 0: km_val *= (1. - d / ar)
                    prev_t = t_ord[i]
                i += 1
            km_out[ei] = km_val
        c_ord = 1. - e_ord; g_val = 1.0; prev_t = -np.inf; i = 0
        for ei, t_ev in enumerate(eval_times_np):
            while i < n and t_ord[i] <= t_ev:
                if t_ord[i] != prev_t:
                    ar = n - i
                    d  = int(np.sum(c_ord[i:][t_ord[i:] == t_ord[i]]))
                    if ar > 0 and d > 0: g_val *= (1. - d / ar)
                    prev_t = t_ord[i]
                i += 1
            g_out[ei] = max(g_val, 1e-4)
        return km_out, g_out

    def forward(self, surv_fn, time, event):
        surv_fn = surv_fn.float()
        t_np    = time.detach().cpu().numpy().astype(np.float64)
        e_np    = event.detach().cpu().numpy().astype(np.float64)
        t_lo, t_hi = np.percentile(t_np, 10), np.percentile(t_np, 90)
        eval_np    = np.linspace(t_lo, t_hi, self.n_eval_pts)
        km_np, g_np = self._km_and_censor(t_np, e_np, eval_np)
        km_t  = torch.tensor(km_np, dtype=torch.float32, device=surv_fn.device)
        ipcw  = torch.tensor(1. / (g_np ** 2 + 1e-8), dtype=torch.float32,
                             device=surv_fn.device)
        ipcw  = ipcw / (ipcw.mean() + 1e-8)
        t_min, t_max = float(t_np.min()), float(t_np.max())
        bfrac = torch.tensor((eval_np - t_min) / (t_max - t_min + 1e-8),
                              dtype=torch.float32, device=surv_fn.device)
        bidxf = (bfrac * (self.K - 1)).clamp(0, self.K - 1)
        lo    = bidxf.long().clamp(0, self.K - 2)
        hi_b  = (lo + 1).clamp(0, self.K - 1)
        alpha = (bidxf - lo.float()).unsqueeze(0)
        pred  = surv_fn[:, lo] * (1 - alpha) + surv_fn[:, hi_b] * alpha
        pmean = pred.mean(dim=0)
        calib = (ipcw * (pmean - km_t) ** 2).mean()
        slope = F.relu(pmean[1:] - pmean[:-1]).mean()
        return calib + 0.1 * slope


class IBSBrierLoss(nn.Module):
    def __init__(self, high_cut_days=365 * 1.5):
        super().__init__()
        self.high_cut = high_cut_days

    def forward(self, surv_fn, time, event, time_bins_tensor):
        surv_fn = surv_fn.float(); time = time.float(); event = event.float()
        B, K    = surv_fn.shape
        t_np    = time.detach().cpu().numpy()
        e_np    = event.detach().cpu().numpy()
        eval_d  = np.unique(np.concatenate([
            np.percentile(t_np, [25, 50, 75]), [self.high_cut]]))
        eval_d  = eval_d[(eval_d > t_np.min()) & (eval_d < np.percentile(t_np, 95))]
        if len(eval_d) == 0:
            return torch.tensor(0., device=surv_fn.device)
        order = np.argsort(t_np)
        t_ord = t_np[order]; e_cen = 1. - e_np[order]
        g_cache = {}
        for t_ev in eval_d:
            gv = 1.; pv = -np.inf
            for ii in range(len(t_ord)):
                if t_ord[ii] > t_ev: break
                if t_ord[ii] != pv:
                    ar = len(t_ord) - ii
                    d  = int(np.sum(e_cen[ii:][t_ord[ii:] == t_ord[ii]]))
                    if ar > 0 and d > 0: gv *= (1. - d / ar)
                    pv = t_ord[ii]
            g_cache[float(t_ev)] = max(gv, 1e-4)
        g_subj = np.ones(B, dtype=np.float32)
        for i in range(B):
            ti    = float(t_np[i])
            cands = [t for t in g_cache if t <= ti]
            g_subj[i] = g_cache[max(cands)] if cands else 1.
        g_st = torch.tensor(g_subj, dtype=torch.float32, device=surv_fn.device)
        total = torch.tensor(0., device=surv_fn.device); n_t = 0
        tmn, tmx = t_np.min(), t_np.max()
        for t_ev in eval_d:
            frac  = (t_ev - tmn) / (tmx - tmn + 1e-8)
            bidxf = torch.tensor(frac * (K - 1), dtype=torch.float32,
                                  device=surv_fn.device).clamp(0, K - 1)
            lo    = bidxf.long().clamp(0, K - 2)
            hi_b  = (lo + 1).clamp(0, K - 1)
            alpha = (bidxf - lo.float())
            St    = surv_fn[:, lo] * (1 - alpha) + surv_fn[:, hi_b] * alpha
            G_t   = g_cache[float(t_ev)]
            t_ev_t= torch.tensor(t_ev, dtype=torch.float32, device=surv_fn.device)
            mev   = ((time <= t_ev_t) & (event > 0.5)).float()
            mce   = (time > t_ev_t).float()
            w_ev  = mev / g_st.clamp(min=1e-4)
            w_ce  = mce / G_t
            total += (w_ev * (0. - St) ** 2 + w_ce * (1. - St) ** 2).mean()
            n_t  += 1
        return total / max(n_t, 1)


class FocalBCE(nn.Module):
    def __init__(self, gamma=2., alpha=0.35):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha

    def forward(self, logits, tgt):
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# --------------------------------------------------------------------------- #
# Shared singleton loss objects                                               #
# --------------------------------------------------------------------------- #
dh_fn    = DeepHitNLL().to(device)
cox_fn   = CoxLoss().to(device)
rank_fn  = PairRankLoss().to(device)
bce_fn   = FocalBCE().to(device)
brier_fn = IBSBrierLoss(high_cut_days=cfg.HIGH_CUT).to(device)
