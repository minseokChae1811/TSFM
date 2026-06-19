"""Minimal inference example for TSFM.

Loads the released TorchScript encoder(s), extracts features from the
bundled CWRU sample STFTs, and evaluates an MLP probe under a LOCO
(leave-one-condition-out) protocol matching the paper.

Protocol:
  - Features: extracted once from frozen encoder.
  - Per fold: hold out one load condition (load_hp ∈ {0,1,2,3}),
    fit MLP on the other three, evaluate on the held-out one.
  - 3 random seeds per fold; reported numbers are mean across folds of
    per-fold (seed-averaged) accuracy.

No model source code is required — each .pt file is a self-contained
TorchScript binary.
"""
import argparse, json, numpy as np, torch
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE     = Path(__file__).parent
DATA_DIR = HERE / "sample_data"
VARIANTS = {
    "ss": HERE / "tsfm_ss_traced.pt",
    "ms": HERE / "tsfm_ms_traced.pt",
}
SEEDS = (42, 0, 1)


def extract_features(model, X, device, batch_size=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).float()\
                          .unsqueeze(1).to(device)
            feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def mlp_loco(feats, y, conds):
    """Leave-one-condition-out MLP probe; 3 seeds per fold."""
    fold_means = []
    for c in np.unique(conds):
        te = conds == c
        tr = ~te
        scaler = StandardScaler().fit(feats[tr])
        Xtr, Xte = scaler.transform(feats[tr]), scaler.transform(feats[te])
        seed_accs = []
        for sd in SEEDS:
            clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                                 random_state=sd).fit(Xtr, y[tr])
            seed_accs.append(clf.score(Xte, y[te]))
        fold_means.append(float(np.mean(seed_accs)))
    return float(np.mean(fold_means)), float(np.std(fold_means)), fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ss", "ms", "both"],
                        default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X     = np.load(DATA_DIR / "cwru_stft.npy")      # (100, 224, 224)
    y     = np.load(DATA_DIR / "cwru_labels.npy")    # (100,) fault
    conds = np.load(DATA_DIR / "cwru_loads.npy")     # (100,) load_hp
    label_map = json.load(open(DATA_DIR / "label_map.json"))
    print(f"Loaded {len(X)} STFTs | {len(set(y))} faults "
          f"× {len(set(conds))} load conditions\n")

    variants = ["ss", "ms"] if args.variant == "both" else [args.variant]
    for v in variants:
        path = VARIANTS[v]
        print(f"--- TSFM-{v.upper()} ({path.name}) ---")
        model = torch.jit.load(str(path), map_location=device).eval()
        feats = extract_features(model, X, device)
        mean, std, fold_means = mlp_loco(feats, y, conds)
        per_fold = "  ".join(f"L{c}={a*100:.1f}"
                              for c, a in zip(sorted(set(conds)), fold_means))
        print(f"  features: {feats.shape}")
        print(f"  per-fold: {per_fold}")
        print(f"  LOCO MLP accuracy (mean ± std over 4 folds): "
              f"{mean*100:.2f} ± {std*100:.2f}%\n")


if __name__ == "__main__":
    main()
