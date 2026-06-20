# TSFM: Spectrogram Foundation Model for Fault Diagnosis

Pretrained encoders for bearing/gear fault diagnosis on STFT spectrograms,
released alongside the paper:

> **"<Paper Title>"** &middot; <Authors> &middot; <Venue> 2026.

This release provides:

1. **Sample data** &mdash; a subset of the CWRU bearing dataset, converted
   to STFT spectrograms in the input format the model expects.
2. **Pretrained model weights** &mdash; two encoders, **TSFM-SS**
   (single-scale) and **TSFM-MS** (multi-scale, the main proposed model),
   shipped as TorchScript checkpoints. CWRU was excluded from the
   pretraining corpus.
3. **Inference example** &mdash; a minimal script that combines the two so
   the encoder can be exercised end-to-end.

Each encoder maps an STFT to a 192-dim feature vector and can be used as
a frozen feature extractor for downstream tasks.

---

## Contents

| Path | Description |
|------|-------------|
| `tsfm_ss_traced.pt` | TorchScript encoder (single-scale variant) |
| `tsfm_ms_traced.pt` | TorchScript encoder (multi-scale variant) |
| `sample_data/cwru_stft.npy` | 240 STFT spectrograms from CWRU |
| `sample_data/cwru_labels.npy` | Fault labels (0..3) |
| `sample_data/cwru_loads.npy` | Load conditions (load_hp ∈ {0,1,2,3}) |
| `sample_data/label_map.json` | Human-readable label names |
| `inference.py` | Minimal example: load → extract → LOCO MLP probe |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn` |

The bundled CWRU sample is a 240-spectrogram subset of the full
evaluation pool used in the paper. The full evaluation set and the
checkpoints for the other held-out datasets are available upon request
to the corresponding authors.

## Quick start

```bash
pip install -r requirements.txt
python inference.py                 # both SS and MS
python inference.py --variant ms    # MS only
```

Expected output:

```
Loaded 240 STFTs | 4 faults × 4 load conditions

--- TSFM-SS (tsfm_ss_traced.pt) ---
  LOCO MLP accuracy (mean ± std over 4 folds): 100.00 ± 0.00%

--- TSFM-MS (tsfm_ms_traced.pt) ---
  LOCO MLP accuracy (mean ± std over 4 folds): 100.00 ± 0.00%
```

## Programmatic use

```python
import torch

model = torch.jit.load("tsfm_ms_traced.pt").eval()
stft = torch.randn(8, 1, 224, 224)     # (B, 1, 224, 224) float32
with torch.no_grad():
    feats = model(stft)                 # (B, 192)
```

Feed `feats` into any downstream classifier of choice.

## Input format

Single-channel STFT magnitude spectrograms shaped `(B, 1, 224, 224)`,
float32.

## Authors

Minseok Chae<sup>a</sup>, Yong Chae Kim<sup>a</sup>, Sang Kyung Lee<sup>a</sup>, Juhyun Kim<sup>a</sup>, Jiheon Kang<sup>a</sup>, Uri Lim<sup>a</sup>, Jong Hyun Choi<sup>a</sup>, Junho Park<sup>a</sup>, Chan Hee Park<sup>b,\*</sup>, Jong Moon Ha<sup>c,\*</sup>, Byeng D. Youn<sup>a,d,e,\*</sup>

<sup>a</sup> Department of Mechanical Engineering, Seoul National University, Seoul 08826, Republic of Korea  
<sup>b</sup> Department of Mechanical Engineering, University of Seoul, Seoul 02504, Republic of Korea  
<sup>c</sup> Department of Mechanical Engineering, Ajou University, Suwon 16499, Republic of Korea  
<sup>d</sup> Institute of Advanced Machines and Design, Seoul National University, Seoul 08826, Republic of Korea  
<sup>e</sup> OnePredict Inc., Seoul 06105, Republic of Korea

Code maintainer (for repository-related questions): Minseok Chae &mdash; `mschae1811@snu.ac.kr`

<sup>\*</sup> Corresponding authors (for research correspondence):
- Chan Hee Park &mdash; `chnypark@uos.ac.kr`
- Jong Moon Ha &mdash; `jmha@ajou.ac.kr`
- Byeng D. Youn &mdash; `bdyoun@snu.ac.kr`

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

> Smith, W. A., & Randall, R. B. (2015). Rolling element bearing
> diagnostics using the Case Western Reserve University data: A
> benchmark study. *Mechanical Systems and Signal Processing*, 64,
> 100-131.

## License

CC BY-NC 4.0 (non-commercial). See `LICENSE`.
