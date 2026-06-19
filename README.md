# TSFM: Spectrogram Foundation Model for Fault Diagnosis

Pretrained encoders for bearing/gear fault diagnosis on STFT spectrograms,
released alongside the paper:

> **"<Paper Title>"** &middot; <Authors> &middot; <Venue> 2026.

This release ships two **inference-ready TorchScript encoders** with a
small labeled sample for sanity checking:

- **TSFM-SS** — single-scale fstrip variant (patch size 4, 56 tokens)
- **TSFM-MS** — multi-scale fstrip variant (patch sizes {2, 16}, 112 tokens,
  cross-scale hybrid fusion). This is the main proposed model in the paper.

Both share the same ViT-Tiny backbone (192-dim, depth 6, 3 heads). They are
intended for reproducibility of the paper's qualitative claims and for
downstream feature-extraction users who do not need to fine-tune or retrain.

---

## Contents

| Path | Description | Size |
|------|-------------|------|
| `tsfm_ss_traced.pt` | TorchScript encoder — single-scale fstrip (ps=4) | ~11 MB |
| `tsfm_ms_traced.pt` | TorchScript encoder — multi-scale fstrip ({2,16}) | ~15 MB |
| `sample_data/cwru_stft.npy` | 100 STFT spectrograms (224×224) from CWRU | ~20 MB |
| `sample_data/cwru_labels.npy` | Integer labels (0..3) | <1 KB |
| `sample_data/label_map.json` | `{0:"normal", 1:"ball", 2:"inner_race", 3:"outer_race"}` | <1 KB |
| `inference.py` | Minimal example: load → extract → KNN evaluate | ~70 lines |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn` | |

The CWRU sample (25 spectrograms per class × 4 classes) is balanced and
randomly drawn from the held-out condition pool used in the paper. The
released checkpoints are the `cwru`-LODO variants — i.e., CWRU was the
held-out dataset during pretraining, so accuracy reported here represents
true zero-shot transfer.

## Quick start

```bash
pip install -r requirements.txt
python inference.py                # both SS and MS
python inference.py --variant ms   # MS only
python inference.py --variant ss   # SS only
```

Expected output (approximately):

```
Loaded 100 STFT samples × 4 classes: {0: 'normal', 1: 'ball', 2: 'inner_race', 3: 'outer_race'}

--- TSFM-SS (tsfm_ss_traced.pt) ---
  features: (100, 192)  (mean=..., std=...)
  KNN-5 accuracy (5-fold): 89.00 ± 8.00%

--- TSFM-MS (tsfm_ms_traced.pt) ---
  features: (100, 192)  (mean=..., std=...)
  KNN-5 accuracy (5-fold): 88.00 ± 11.22%
```

(Numbers on the bundled 100-sample subset are noisy; the paper reports
full-test-set LOCO macro accuracy across 13 datasets, where TSFM-MS
outperforms TSFM-SS by ~2 percentage points.)

## Programmatic use

Each model accepts a `(B, 1, 224, 224)` float32 STFT tensor and returns a
`(B, 192)` feature vector (mean-pooled over tokens).

```python
import torch

model = torch.jit.load("tsfm_ms_traced.pt").eval()
stft = torch.randn(8, 1, 224, 224)        # your STFT batch
with torch.no_grad():
    feats = model(stft)                    # (8, 192)
```

Downstream usage (anomaly scoring, KNN, MLP probe, etc.) is the user's
responsibility — feed `feats` into any classifier of choice.

## Input convention

Inputs must be 224×224 single-channel STFT magnitude spectrograms with the
**frequency axis vertical and the time axis horizontal**. We do not ship
the preprocessing pipeline; users supplying their own data should ensure:

- Sampling rate is normalized to the analysis range used in the paper
- STFT window/hop produce a 224-frame time × 224-bin frequency representation
- Values are float32 magnitude (not log-magnitude) and unit-normalized per sample

## What this release does **not** include

To protect ongoing research and lab know-how, the following are NOT released:

- Model source code (encoders are shipped as TorchScript binaries only)
- Pretraining code (MAE objective, masking, decoder)
- 1D-signal → STFT preprocessing pipeline
- Multi-dataset training scripts and ablation configs
- Fine-tuning recipes

For collaborations or licensing inquiries, please contact the
corresponding author.

## Citation

```bibtex
@article{tsfm_2026,
  title   = {<Paper Title>},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {2026},
}
```

When using the bundled CWRU samples, please also cite:

> Case Western Reserve University Bearing Data Center.
> https://engineering.case.edu/bearingdatacenter

## License

CC BY-NC 4.0 (non-commercial). See `LICENSE`. Commercial use requires a
separate agreement with the corresponding author.
