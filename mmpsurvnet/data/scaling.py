"""
Per-fold feature scaling.

All normalisation statistics (age mean/std, radiomics 1st/99th percentile clip
bounds, StandardScaler) are fit on the training fold and applied unchanged to
the test fold.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..config import RAD_COLS, FEAT_COLS


def build_fold_scaler(df_train_raw, df_test_raw):
    # Statistics computed on TRAINING FOLD only
    age_mu = df_train_raw['age_raw'].mean()
    age_sg = df_train_raw['age_raw'].std() + 1e-8
    scaler = StandardScaler()

    # Per-fold radiomics clipping: 1st/99th percentile computed on train fold.
    # The same clip bounds are applied to the test fold (no re-fitting).
    rad_clips = {}
    for col in RAD_COLS:
        lo = df_train_raw[col].replace([np.inf, -np.inf], np.nan).quantile(0.01)
        hi = df_train_raw[col].replace([np.inf, -np.inf], np.nan).quantile(0.99)
        rad_clips[col] = (lo, hi)

    def scale(df_raw, fit=False):
        df = df_raw.copy()
        # Normalise age using train-fold mu/sigma
        df['age_norm'] = (df['age_raw'] - age_mu) / age_sg
        # Clip radiomics to train-fold 1st-99th percentile range
        for col, (lo, hi) in rad_clips.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        fc = [c for c in FEAT_COLS if c in df.columns]
        X  = (df[fc].replace([np.inf, -np.inf], np.nan)
                     .fillna(0).values.astype(np.float32))
        if fit:
            scaler.fit(X)    # fit on train only
        Xs = scaler.transform(X)
        for j, col in enumerate(fc):
            df[col] = Xs[:, j]
        return df

    df_tr = scale(df_train_raw, fit=True)    # fit + transform train
    df_te = scale(df_test_raw,  fit=False)   # transform test (no refit)
    return df_tr, df_te, age_mu, age_sg
