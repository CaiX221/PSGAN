from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_dir = Path("/DATA/yantongliu/PSGAN-ours/PSGAN-spiga/results")
    summary_csv = results_dir / "nme_comparison_summary.csv"
    out_png = results_dir / "nme_comparison_visualization.png"

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
    method_labels = [display_names[m] for m in methods]

    full_mean = np.array([float(rows_by_method[m]["nme_full_mean"]) for m in methods], dtype=np.float32)
    full_std = np.array([float(rows_by_method[m]["nme_full_std"]) for m in methods], dtype=np.float32)

    local_metrics = [
        "nme_jaw_mean",
        "nme_brow_mean",
        "nme_nose_mean",
        "nme_eyes_mean",
        "nme_mouth_mean",
    ]
    local_labels = ["Jaw", "Brow", "Nose", "Eyes", "Mouth"]
    local_values = np.vstack(
        [
            np.array([float(rows_by_method[m][metric]) for m in methods], dtype=np.float32)
            for metric in local_metrics
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    # Left: NME Full with std error bars.
    ax = axes[0]
    colors = ["#4C78A8", "#59A14F", "#E15759"]
    bars = ax.bar(method_labels, full_mean, yerr=full_std, capsize=6, color=colors)
    ax.set_title("NME Full (Lower is Better)")
    ax.set_ylabel("NME")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b, mean_val in zip(bars, full_mean):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9)

    # Right: Local-region grouped bar chart.
    ax2 = axes[1]
    x = np.arange(len(local_labels))
    width = 0.24
    for i, method in enumerate(methods):
        ax2.bar(
            x + (i - 1) * width,
            local_values[:, i],
            width=width,
            label=display_names[method],
            color=colors[i],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(local_labels)
    ax2.set_title("Local NME by Facial Region")
    ax2.set_ylabel("NME")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    plt.suptitle("Landmark Structure Error Comparison (SPIGA 68pt)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved visualization: {out_png}")


if __name__ == "__main__":
    main()
