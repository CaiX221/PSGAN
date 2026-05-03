import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_root = Path("/DATA/yantongliu/PSGAN-ours/PSGAN-spiga/results")
    summary_csv = results_root / "output" / "deltae_regional_summary.csv"
    out_png = results_root / "output" / "deltae_regional_visualization.png"

    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows_by_method = {r["method"]: r for r in rows}
    order = ["baseline", "ours", "stablemakeup"]
    methods = [m for m in order if m in rows_by_method]
    display_names = {
        "baseline": "PSGAN",
        "ours": "ours",
        "stablemakeup": "Stable-Makeup",
    }

    labels = ["Lip", "Eye", "Skin"]
    metrics = ["deltae_lip_mean", "deltae_eye_mean", "deltae_skin_mean"]
    values = np.vstack(
        [
            np.array([float(rows_by_method[m][metric]) for m in methods], dtype=np.float32)
            for metric in metrics
        ]
    )
    avg_vals = np.array([float(rows_by_method[m]["deltae_avg_mean"]) for m in methods], dtype=np.float32)
    avg_stds = np.array([float(rows_by_method[m]["deltae_avg_std"]) for m in methods], dtype=np.float32)

    colors = ["#4C78A8", "#59A14F", "#E15759"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    ax1 = axes[0]
    x = np.arange(len(labels))
    width = 0.24
    for i, m in enumerate(methods):
        ax1.bar(
            x + (i - 1) * width,
            values[:, i],
            width=width,
            color=colors[i],
            label=display_names[m],
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Regional DeltaE00")
    ax1.set_ylabel("DeltaE00")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend()

    ax2 = axes[1]
    method_labels = [display_names[m] for m in methods]
    bars = ax2.bar(method_labels, avg_vals, yerr=avg_stds, capsize=6, color=colors)
    ax2.set_title("Average DeltaE00 (Lower is Better)")
    ax2.set_ylabel("DeltaE00")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, avg_vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Makeup Color Transfer Accuracy (CIEDE2000)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved visualization: {out_png}")


if __name__ == "__main__":
    main()
