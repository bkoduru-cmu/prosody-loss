"""
Linear Probe - Emotion Classification per Layer
Works on hidden_states.jsonl produced by either extract_states_*.py

Usage:
    python probe.py --states probe_states_omni/hidden_states.jsonl --title "Qwen2.5-Omni"
    python probe.py --states probe_states_qwen2audio/hidden_states.jsonl --title "Qwen2-Audio"
    python probe.py --states probe_states_omni/hidden_states.jsonl --speaker_split   # speaker-held-out CV
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_states(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} clips from {path}")
    return records


# ─────────────────────────────────────────────
# BUILD LAYER MATRICES
# ─────────────────────────────────────────────
def build_layer_data(records):
    """
    Returns:
        layers: dict of layer_name → np.array (n_clips, hidden_dim)
        labels: np.array of int emotion labels (n_clips,)
        speakers: np.array of speaker strings (n_clips,)
        label_names: list of emotion names
    """
    emotions  = [r["emotion"] for r in records]
    speakers  = np.array([r["speaker"] for r in records])
    le        = LabelEncoder()
    labels    = le.fit_transform(emotions)
    label_names = list(le.classes_)

    # Infer layer count from first record
    r0      = records[0]
    n_enc   = len(r0["enc"])
    n_dec   = len(r0["dec"])
    has_prj = r0["prj"] is not None

    layer_names = (
        [f"enc_{i}" for i in range(n_enc)]
        + (["projector"] if has_prj else [])
        + [f"dec_{i}" for i in range(n_dec)]
    )

    layers = {}
    for name in layer_names:
        if name.startswith("enc_"):
            idx = int(name.split("_")[1])
            mat = np.array([r["enc"][idx] for r in records], dtype=np.float32)
        elif name == "projector":
            mat = np.array([r["prj"] for r in records], dtype=np.float32)
        else:
            idx = int(name.split("_")[1])
            mat = np.array([r["dec"][idx] for r in records], dtype=np.float32)
        layers[name] = mat

    print(f"Layers      : {len(layer_names)}")
    print(f"Encoder     : {n_enc} layers, dim={records[0]['enc'][0].__len__()}")
    print(f"Decoder     : {n_dec} layers, dim={records[0]['dec'][0].__len__()}")
    print(f"Classes     : {label_names}")
    from collections import Counter
    dist = Counter(emotions)
    print("Distribution:")
    for e, n in sorted(dist.items()):
        print(f"  {e:15s}: {n}")

    return layers, labels, speakers, label_names, layer_names


# ─────────────────────────────────────────────
# PROBE
# ─────────────────────────────────────────────
def probe_layer(X, y, groups=None, speaker_split=False, n_folds=5):
    """
    Trains a logistic regression probe on X → y.
    If speaker_split=True, uses leave-one-speaker-out CV.
    Otherwise uses StratifiedKFold.
    Returns mean accuracy and std.
    """
    scaler = StandardScaler()
    clf    = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 multi_class="multinomial")

    if speaker_split and groups is not None:
        cv      = LeaveOneGroupOut()
        splits  = list(cv.split(X, y, groups))
    else:
        cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits  = list(cv.split(X, y))

    accs = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, clf.predict(X_test)))

    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",       required=True,
                        help="Path to hidden_states.jsonl")
    parser.add_argument("--title",        default=None,
                        help="Plot title")
    parser.add_argument("--output",       default=None,
                        help="Output PNG path")
    parser.add_argument("--speaker_split", action="store_true",
                        help="Use leave-one-speaker-out CV instead of StratifiedKFold")
    parser.add_argument("--folds",        type=int, default=5,
                        help="Number of folds for StratifiedKFold (default 5)")
    args = parser.parse_args()

    records = load_states(args.states)
    layers, labels, speakers, label_names, layer_names = build_layer_data(records)

    n_classes  = len(label_names)
    chance     = 1.0 / n_classes
    cv_type    = "leave-one-speaker-out" if args.speaker_split else f"{args.folds}-fold stratified"

    print(f"\nCV strategy : {cv_type}")
    print(f"Chance level: {chance:.3f} ({n_classes} classes)")
    print(f"\nProbing {len(layer_names)} layers...")

    results = {}   # layer_name → (mean_acc, std_acc)

    for name in tqdm(layer_names, desc="Probing"):
        X      = layers[name]
        groups = speakers if args.speaker_split else None
        mean, std = probe_layer(X, labels, groups=groups,
                                speaker_split=args.speaker_split,
                                n_folds=args.folds)
        results[name] = (mean, std)
        tqdm.write(f"  {name:12s}: {mean:.3f} ± {std:.3f}")

    # Save results JSON
    out_dir  = os.path.dirname(args.states)
    json_out = os.path.join(out_dir, "probe_results.json")
    with open(json_out, "w") as f:
        json.dump({
            "layer_names":   layer_names,
            "label_names":   label_names,
            "cv_type":       cv_type,
            "chance":        chance,
            "results": {k: {"mean": v[0], "std": v[1]} for k, v in results.items()},
        }, f, indent=2)
    print(f"\n💾 Results saved: {json_out}")

    # ─────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────
    means = [results[n][0] for n in layer_names]
    stds  = [results[n][1] for n in layer_names]
    xs    = list(range(len(layer_names)))

    n_enc   = sum(1 for n in layer_names if n.startswith("enc_"))
    has_prj = "projector" in layer_names
    proj_idx = n_enc  # index of projector (or first dec layer if no proj)

    plt.figure(figsize=(14, 6))
    plt.plot(xs, means, color="steelblue", linewidth=2, label="Probe accuracy")
    plt.fill_between(xs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color="steelblue")
    plt.axhline(y=chance, color="gray", linestyle=":", linewidth=1.5,
                label=f"Chance ({chance:.2f})")
    plt.axvline(x=proj_idx, color="red", linestyle="--", linewidth=2,
                label="Projector")
    plt.axvspan(0, proj_idx - 0.5, color="blue",  alpha=0.05, label="Encoder")
    plt.axvspan(proj_idx + 0.5, len(layer_names), color="green", alpha=0.05,
                label="LLM Decoder")

    auto_title = args.title or "Linear Probe – Emotion Classification per Layer"
    plt.title(f"{auto_title}  [{cv_type}]", fontweight="bold", fontsize=13)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if args.output is None:
        args.output = os.path.join(out_dir, "probe_plot.png")
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {args.output}")

    # Print top 5 layers
    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    print(f"\nTop 5 layers by accuracy:")
    for name, (mean, std) in ranked[:5]:
        print(f"  {name:12s}: {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
