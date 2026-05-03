import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_pairs(pairs_file: Path):
    pairs = []
    with pairs_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def resolve_stable_output(stable_dir: Path, pair_id: int):
    matches = sorted(stable_dir.glob(f"pair{pair_id:02d}_*.png"))
    return matches[0] if matches else None


def compute_iou_and_dice(mask_a: np.ndarray, mask_b: np.ndarray):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    sum_ab = mask_a.sum() + mask_b.sum()
    if union == 0 or sum_ab == 0:
        return None, None
    iou = float(inter / union)
    dice = float((2.0 * inter) / sum_ab)
    return iou, dice


def summarize(values):
    if len(values) == 0:
        return "", "", 0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def main():
    parser = argparse.ArgumentParser(description="Evaluate makeup-region overlap (IoU/Dice) between output and reference.")
    parser.add_argument("--project_root", default="/DATA/yantongliu/PSGAN-ours/PSGAN-spiga")
    parser.add_argument("--pairs_file", default="assets/images/pairs.txt")
    parser.add_argument("--reference_dir", default="assets/images/makeup")
    parser.add_argument("--baseline_dir", default="results/baseline_updated30")
    parser.add_argument("--ours_dir", default="results/ours_updated30")
    parser.add_argument("--stable_dir", default="results/stablemakeup_updated30")
    parser.add_argument("--output_prefix", default="results/output/makeup_overlap")
    parser.add_argument("--parser_device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    import faceutils as futils

    def resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (project_root / p)

    pairs_file = resolve_path(args.pairs_file)
    reference_dir = resolve_path(args.reference_dir)
    baseline_dir = resolve_path(args.baseline_dir)
    ours_dir = resolve_path(args.ours_dir)
    stable_dir = resolve_path(args.stable_dir)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(pairs_file)
    face_parser = futils.mask.FaceParser(device=args.parser_device)

    # Label sets are based on this project's parser mapping:
    # lip region values commonly appear as 7/8/9 in parsed maps.
    # eye-related makeup area is approximated with brow/eye labels 2/3/4/5.
    lip_labels = {7, 8, 9}
    eye_labels = {2, 3, 4, 5}

    methods = ["baseline", "ours", "stablemakeup"]
    metric_names = [
        "iou_lip",
        "dice_lip",
        "iou_eye",
        "dice_eye",
        "iou_makeup",
        "dice_makeup",
    ]
    metric_store = {m: {k: [] for k in metric_names} for m in methods}
    rows = []
    ref_mask_cache = {}

    for pair_id, (_, ref_name) in enumerate(pairs, start=1):
        ref_path = reference_dir / ref_name
        outputs = {
            "baseline": baseline_dir / f"pair{pair_id:02d}_result.png",
            "ours": ours_dir / f"pair{pair_id:02d}_result.png",
            "stablemakeup": resolve_stable_output(stable_dir, pair_id),
        }

        if not ref_path.exists():
            for method in methods:
                rows.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "status": "missing_reference",
                        "reference_path": str(ref_path),
                        "output_path": "",
                    }
                )
            continue

        if pair_id not in ref_mask_cache:
            ref_img = Image.open(ref_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            ref_parse = face_parser.parse(np.asarray(ref_img)).squeeze().cpu().numpy().astype(np.int32)
            ref_lip = np.isin(ref_parse, list(lip_labels))
            ref_eye = np.isin(ref_parse, list(eye_labels))
            ref_makeup = np.logical_or(ref_lip, ref_eye)
            ref_mask_cache[pair_id] = (ref_lip, ref_eye, ref_makeup)

        ref_lip, ref_eye, ref_makeup = ref_mask_cache[pair_id]

        for method in methods:
            out_path = outputs[method]
            row = {
                "pair_id": pair_id,
                "method": method,
                "status": "ok",
                "reference_path": str(ref_path),
                "output_path": str(out_path) if out_path else "",
            }

            if out_path is None or not out_path.exists():
                row["status"] = "missing_output"
                rows.append(row)
                continue

            try:
                out_img = Image.open(out_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                out_parse = face_parser.parse(np.asarray(out_img)).squeeze().cpu().numpy().astype(np.int32)
            except Exception as e:
                row["status"] = f"parse_error:{e}"
                rows.append(row)
                continue

            out_lip = np.isin(out_parse, list(lip_labels))
            out_eye = np.isin(out_parse, list(eye_labels))
            out_makeup = np.logical_or(out_lip, out_eye)

            iou_lip, dice_lip = compute_iou_and_dice(out_lip, ref_lip)
            iou_eye, dice_eye = compute_iou_and_dice(out_eye, ref_eye)
            iou_m, dice_m = compute_iou_and_dice(out_makeup, ref_makeup)

            if None in [iou_lip, dice_lip, iou_eye, dice_eye, iou_m, dice_m]:
                row["status"] = "empty_region"
                rows.append(row)
                continue

            row.update(
                {
                    "iou_lip": iou_lip,
                    "dice_lip": dice_lip,
                    "iou_eye": iou_eye,
                    "dice_eye": dice_eye,
                    "iou_makeup": iou_m,
                    "dice_makeup": dice_m,
                }
            )
            rows.append(row)

            metric_store[method]["iou_lip"].append(iou_lip)
            metric_store[method]["dice_lip"].append(dice_lip)
            metric_store[method]["iou_eye"].append(iou_eye)
            metric_store[method]["dice_eye"].append(dice_eye)
            metric_store[method]["iou_makeup"].append(iou_m)
            metric_store[method]["dice_makeup"].append(dice_m)

    per_pair_csv = output_prefix.with_name(output_prefix.name + "_per_pair.csv")
    with per_pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "method",
                "status",
                "reference_path",
                "output_path",
                "iou_lip",
                "dice_lip",
                "iou_eye",
                "dice_eye",
                "iou_makeup",
                "dice_makeup",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in methods:
        row = {"method": method}
        for metric in metric_names:
            mean, std, n = summarize(metric_store[method][metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_n"] = n
        summary_rows.append(row)

    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"]
        for metric in metric_names:
            fieldnames += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-pair metrics: {per_pair_csv}")
    print(f"Saved summary: {summary_csv}")
    print("\n=== Overlap metrics (higher is better) ===")
    for r in summary_rows:
        print(
            f"{r['method']:12s} "
            f"IoU_makeup={r['iou_makeup_mean']} Dice_makeup={r['dice_makeup_mean']} "
            f"IoU_lip={r['iou_lip_mean']} IoU_eye={r['iou_eye_mean']}"
        )


if __name__ == "__main__":
    main()
