# TSFM: Spectrogram Foundation Model for Fault Diagnosis

Pretrained encoders for bearing/gear fault diagnosis on STFT spectrograms,
released alongside the paper:

> **"<Paper Title>"** &middot; <Authors> &middot; <Venue> 2026.

This release ships two inference-ready TorchScript encoders, **TSFM-SS**
(single-scale) and **TSFM-MS** (multi-scale, the main proposed model),
together with a small labeled sample for sanity checking. Both produce a
192-dim feature vector per spectrogram and can be used directly as
frozen feature extractors for downstream tasks.

---

## Contents

| Path | Description |
|------|-------------|
| `tsfm_ss_traced.pt` | TorchScript encoder (single-scale variant) |
| `tsfm_ms_traced.pt` | TorchScript encoder (multi-scale variant) |
| `sample_data/cwru_stft.npy` | 100 STFT spectrograms from CWRU |
| `sample_data/cwru_labels.npy` | Fault labels (0..3) |
| `sample_data/cwru_loads.npy` | Load conditions (load_hp ∈ {0,1,2,3}) |
| `sample_data/label_map.json` | Human-readable label names |
| `inference.py` | Minimal example: load → extract → LOCO MLP probe |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn` |

The CWRU sample (25 spectrograms × 4 fault classes, balanced across 4
load conditions) supports the same leave-one-condition-out (LOCO)
evaluation protocol used in the paper. The released checkpoints were
pretrained with CWRU held out, so reported accuracy represents true
zero-shot transfer.

## Quick start

```bash
pip install -r requirements.txt
python inference.py                 # both SS and MS
python inference.py --variant ms    # MS only
```

Expected output:

```
Loaded 100 STFTs | 4 faults × 4 load conditions

--- TSFM-SS (tsfm_ss_traced.pt) ---
  LOCO MLP accuracy (mean ± std over 4 folds): 95.31 ± 3.11%

--- TSFM-MS (tsfm_ms_traced.pt) ---
  LOCO MLP accuracy (mean ± std over 4 folds): 96.92 ± 3.44%
```

(Numbers on the bundled 100-sample subset are necessarily noisy; the
paper reports full-test-set LOCO macro accuracy across 13 datasets.)

## Programmatic use

```python
import torch

model = torch.jit.load("tsfm_ms_traced.pt").eval()
stft = torch.randn(8, 1, 224, 224)     # (B, 1, 224, 224) float32
with torch.no_grad():
    feats = model(stft)                 # (B, 192)
```

Feed `feats` into any downstream classifier of choice (MLP, KNN, anomaly
scorer, etc.).

## Input format

Single-channel STFT magnitude spectrograms shaped `(B, 1, 224, 224)`,
float32.

## Citation

```bibtex
@article{tsfm_2026,
  title   = {<Paper Title>},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {2026},
}
```

When using the bundled CWRU samples, please also cite the
Case Western Reserve University Bearing Data Center.

## License

CC BY-NC 4.0 (non-commercial). See `LICENSE`.
