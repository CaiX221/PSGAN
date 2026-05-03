import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_root = Path("/DATA/yantongliu/PSGAN-ours/PSGAN-spiga/results")
    summary_csv = results_root / "output" / "makeup_overlap_summary.csv"
    out_png = results_root / "output" / "makeup_overlap_visualization.png"

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

    iou_makeup = np.array([float(rows_by_method[m]["iou_makeup_mean"]) for m in methods], dtype=np.float32)
    iou_makeup_std = np.array([float(rows_by_method[m]["iou_makeup_std"]) for m in methods], dtype=np.float32)
    dice_makeup = np.array([float(rows_by_method[m]["dice_makeup_mean"]) for m in methods], dtype=np.float32)
    dice_makeup_std = np.array([float(rows_by_method[m]["dice_makeup_std"]) for m in methods], dtype=np.float32)

    region_labels = ["Lip IoU", "Eye IoU", "Lip Dice", "Eye Dice"]
    region_metrics = ["iou_lip_mean", "iou_eye_mean", "dice_lip_mean", "dice_eye_mean"]
    region_vals = np.vstack(
        [
            np.array([float(rows_by_method[m][metric]) for m in methods], dtype=np.float32)
            for metric in region_metrics
        ]
    )

    colors = ["#4C78A8", "#59A14F", "#E15759"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    ax1 = axes[0]
    x = np.arange(2)
    width = 0.24
    for i, m in enumerate(methods):
        bars = ax1.bar(
            x + (i - 1) * width,
            [iou_makeup[i], dice_makeup[i]],
            width=width,
            yerr=[iou_makeup_std[i], dice_makeup_std[i]],
            capsize=5,
            color=colors[i],
            label=display_names[m],
        )
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Makeup IoU", "Makeup Dice"])
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Overall Makeup Overlap (Higher is Better)")
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend()

    ax2 = axes[1]
    x2 = np.arange(len(region_labels))
    for i, m in enumerate(methods):
        ax2.bar(
            x2 + (i - 1) * width,
            region_vals[:, i],
            width=width,
            color=colors[i],
            label=display_names[m],
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(region_labels)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Regional Overlap")
    ax2.set_ylabel("Score")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle("Makeup Region IoU / Dice Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved visualization: {out_png}")


if __name__ == "__main__":
    main()
