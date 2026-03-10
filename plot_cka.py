"""
Plot CKA results from a saved global_results.json.

Usage:
    python plot_cka.py --results cka_omni_output/global_results.json
    python plot_cka.py --results cka_qwen2audio_output/global_results.json --title "Qwen2-Audio"
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cka(results_path, title=None, output_path=None):
    with open(results_path) as f:
        gd = json.load(f)

    global_agg     = gd["aggregated_stats"]
    total_processed = gd["total_processed"]
    total_pairs     = gd["total_pairs"]

    if not global_agg:
        print("No data to plot yet.")
        return

    # Infer layer order from first pair
    sample_p = list(global_agg.keys())[0]
    enc_ks = sorted([k for k in global_agg[sample_p] if k.startswith("enc_")],
                    key=lambda x: int(x.split("_")[-1]))
    dec_ks = sorted([k for k in global_agg[sample_p] if k.startswith("dec_")],
                    key=lambda x: int(x.split("_")[-1]))
    has_proj = "projector" in global_agg[sample_p]
    all_ks   = enc_ks + (["projector"] if has_proj else []) + dec_ks
    proj_idx = len(enc_ks)

    print(f"Encoder layers : {len(enc_ks)}")
    print(f"Decoder layers : {len(dec_ks)}")
    print(f"Has projector  : {has_proj}")
    print(f"Style pairs    : {len(global_agg)}")
    print(f"Pairs processed: {total_processed}/{total_pairs} "
          f"({100*total_processed/total_pairs:.1f}%)")

    plt.figure(figsize=(14, 7))

    for pair, layers in global_agg.items():
        y = [layers[k]["mean"] for k in all_ks if k in layers]
        plt.plot(y, label=pair.replace("-", " v "), alpha=0.7, linewidth=1)

    plt.axvline(x=proj_idx, color="red", linestyle="--", linewidth=2, label="Projector")
    plt.axvspan(0, proj_idx - 0.5, color="blue",  alpha=0.05, label="Encoder")
    plt.axvspan(proj_idx + 0.5, len(all_ks), color="green", alpha=0.05, label="LLM Decoder")

    auto_title = title or f"CKA Drift (n={total_processed} pairs)"
    plt.title(auto_title, fontweight="bold", fontsize=14)
    plt.ylabel("CKA Score", fontsize=12)
    plt.xlabel("Layer Index", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(results_path), "plot.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to global_results.json")
    parser.add_argument("--title",   default=None,
                        help="Optional plot title override")
    parser.add_argument("--output",  default=None,
                        help="Output PNG path (default: plot.png next to results file)")
    args = parser.parse_args()

    plot_cka(args.results, title=args.title, output_path=args.output)
