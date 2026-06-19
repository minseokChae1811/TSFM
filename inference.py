"""Minimal inference example for TSFM.

Loads the released TorchScript encoder(s) — TSFM-SS (single-scale fstrip,
ps=4) and TSFM-MS (multi-scale fstrip, ms_scales={2,16}) — extracts
features from the bundled CWRU sample STFTs, and reports 5-fold KNN
accuracy as a sanity check.

No model source code is required — each .pt file is a self-contained
TorchScript binary.
"""
import argparse, json, numpy as np, torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

HERE     = Path(__file__).parent
DATA_DIR = HERE / "sample_data"
VARIANTS = {
    "ss": HERE / "tsfm_ss_traced.pt",
    "ms": HERE / "tsfm_ms_traced.pt",
}


def extract_features(model, X, device, batch_size=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).float()\
                          .unsqueeze(1).to(device)
            feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def knn_5fold(feats, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(feats, y):
        knn = KNeighborsClassifier(n_neighbors=5).fit(feats[tr], y[tr])
        accs.append(knn.score(feats[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ss", "ms", "both"],
                        default="both",
                        help="Which TorchScript model to run.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(DATA_DIR / "cwru_stft.npy")
    y = np.load(DATA_DIR / "cwru_labels.npy")
    label_map = json.load(open(DATA_DIR / "label_map.json"))
    print(f"Loaded {len(X)} STFT samples × {len(set(y))} classes: "
          f"{ {int(k): v for k, v in label_map.items()} }\n")

    variants = ["ss", "ms"] if args.variant == "both" else [args.variant]
    for v in variants:
        path = VARIANTS[v]
        print(f"--- TSFM-{v.upper()} ({path.name}) ---")
        model = torch.jit.load(str(path), map_location=device).eval()
        feats = extract_features(model, X, device)
        mean, std = knn_5fold(feats, y)
        print(f"  features: {feats.shape}  "
              f"(mean={feats.mean():.3f}, std={feats.std():.3f})")
        print(f"  KNN-5 accuracy (5-fold): {mean*100:.2f} ± {std*100:.2f}%\n")


if __name__ == "__main__":
    main()
