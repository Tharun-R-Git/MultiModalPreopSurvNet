# MultiModalPreopSurvNet

Preoperative multimodal deep-learning model for **overall-survival prediction in
glioblastoma** on the **UCSF-PDGM** cohort. The model fuses three preoperative
information sources:

- **MRI** — FLAIR + T1-contrast-enhanced (T1ce) volumes via a 3-D ResNet encoder
  with attention-based survival pooling;
- **Preoperative clinical** — sex and age (2 features);
- **Radiomics** — 33 shape / first-order / texture (GLCM, GLRLM, GLSZM) features
  extracted with [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics).

Fusion is performed by a **Cross-Modal Attention Fusion (CMAF)** module with a
learnable image/clinical gate. Training uses a composite survival objective
(discrete-time NLL + Cox + pairwise ranking + IPCW calibration + IPCW Brier +
focal-BCE risk) plus L1 feature selection on the radiomic weights, with masked
autoencoder (MMAP) self-supervised pretraining of the image encoder. The model
is evaluated under a **5-fold stratified cross-validation** protocol.

---

## Repository structure

```
MultiModalPreopSurvNet/
├── mmpsurvnet/                   # Installable Python package
│   ├── config.py                 # Cfg hyper-parameters, paths, feature lists, device
│   ├── losses.py                 # Discrete-time / Cox / Rank / Calib / Brier / FocalBCE
│   ├── metrics.py                # C-index (IPCW), IBS, Brier@18m, isotonic recal, evaluate_fold
│   ├── train.py                  # Per-fold training loop (get_lr_factor, train_one_fold)
│   ├── cross_validation.py       # 5-fold CV driver (run_5fold_cv)
│   ├── evaluate.py               # Pooled OOF metrics + bootstrap 95% CI
│   ├── reporting.py              # Results table, 8-panel figure, results JSON
│   ├── data/
│   │   ├── metadata.py           # load_metadata, discover_patients, NIfTI verification
│   │   ├── radiomics.py          # PyRadiomics extractor (+ NumPy fallback), caching
│   │   ├── scaling.py            # build_fold_scaler
│   │   └── dataset.py            # load_vol, load_seg_mask, GliomaDS
│   └── models/
│       ├── layers.py             # ConvBnSilu, SE3D, SpatAttn3D, ResBlock3D, SurvivalPool, CMAF
│       ├── network.py            # MultiModalPreopSurvNet architecture
│       └── mmap.py               # MaskedMAP pretraining
├── scripts/
│   └── run_experiment.py         # End-to-end entry point
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone <your-repo-url>
cd MultiModalPreopSurvNet

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # install the mmpsurvnet package (editable)
```

A CUDA-capable GPU is strongly recommended. If PyRadiomics is unavailable, the
pipeline falls back to a reduced NumPy radiomics implementation (not recommended
for reproducing the reported results).

---

## Data

The model expects the **UCSF-PDGM** dataset:

- a metadata CSV with at least the columns `OS`, `1-dead 0-alive`, `Sex`,
  `Age at MRI`, and a patient identifier;
- per-patient folders containing FLAIR, T1ce and tumour-segmentation NIfTI
  volumes (baseline studies; follow-up `_FU` studies are ignored).

Update the paths in `mmpsurvnet/config.py` (`Cfg.CSV_PATH`, `Cfg.MRI_ROOT`,
`Cfg.OUT_DIR`, …) to point to your local copy. The defaults match a Kaggle
environment.

---

## Usage

Run the full experiment (data loading → radiomics → 5-fold CV → pooled metrics →
figure + JSON):

```bash
cd MultiModalPreopSurvNet
python -m scripts.run_experiment
```

Outputs (paths configurable in `Cfg`):

- `mmps_best_models/best_fold{1..5}.pth` — best per-fold checkpoints
- `mmps_mmap_fold{1..5}.pth` — per-fold MMAP encoder weights
- `mmps_radiomics_cache.pkl` — cached radiomic features
- `mmps_results.png` — 8-panel publication figure
- `mmps_results.json` — per-fold and pooled metrics with bootstrap 95% CIs

### Using components programmatically

```python
from mmpsurvnet.config import cfg, RAD_COLS
from mmpsurvnet.data import load_metadata, discover_patients, extract_radiomics
from mmpsurvnet.cross_validation import run_5fold_cv
from mmpsurvnet.evaluate import compute_pooled_metrics

df_all              = load_metadata(cfg.CSV_PATH)
df_valid, all_paths = discover_patients(df_all, cfg.MRI_ROOT)
rad_df              = extract_radiomics(df_valid, all_paths)
for c in RAD_COLS:
    df_valid[c] = rad_df[c].values

fold_results, oof_records, ref_os, ref_ev, gate_hist = run_5fold_cv(df_valid, all_paths)
pooled = compute_pooled_metrics(oof_records, ref_os, ref_ev)
```

---

## Cross-validation protocol

The cohort is split with `StratifiedKFold` on a survival/event stratification
label built from fixed OS thresholds (180 / 365 / 548 days). For every fold, the
following are derived from the **training fold only** and then applied to the
held-out fold:

1. **Feature scaling** — `StandardScaler`, age mean/std and radiomics
   1st/99th-percentile clip bounds (`build_fold_scaler`).
2. **Time bins** — discrete-time survival bins from training-fold OS percentiles.
3. **MMAP pretraining** — masked-autoencoder pretraining of the image encoder.
4. **Isotonic recalibration** — per-bin IPCW-weighted calibrators.

---

## Reported metrics

- **Harrell's C** and **IPCW time-dependent C-index** (`Cdt`)
- **Integrated Brier Score** (IBS) and **18-month Brier score**
- **AUC / sensitivity / specificity / F1** for 18-month high-risk classification
- **Log-rank p-value** for the median-risk survival split
- Bootstrap **95% confidence intervals** (n = 500) for all pooled metrics

---

## Notes

- `mmpsurvnet/config.py` is the single source of truth for every hyper-parameter
  and path; nothing is hard-coded elsewhere.
- The NumPy radiomics fallback in `data/radiomics.py` is retained for
  environments without PyRadiomics — install PyRadiomics for reproducible
  results.

## Citation

If you use this code, please cite the accompanying paper. _(Citation / BibTeX to
be added on acceptance.)_

## License

Released under the [MIT License](LICENSE).
