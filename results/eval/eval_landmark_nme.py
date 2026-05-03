import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from facelib import FaceDetector
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


LOCAL_REGIONS = {
    "jaw": list(range(0, 17)),
    "brow": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "mouth": list(range(48, 68)),
}


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


def resolve_stable_path(stable_dir: Path, pair_id: int):
    matches = sorted(stable_dir.glob(f"pair{pair_id:02d}_*.png"))
    return matches[0] if matches else None


def detect_landmarks_68(image_path: Path, detector: FaceDetector, processor: SPIGAFramework):
    image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    _, boxes, _, _ = detector.detect_align(image_cv)
    if boxes is None:
        return None

    boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
    if len(boxes_np) == 0:
        return None

    box_ls = []
    for box in boxes_np:
        x0, y0, x1, y1 = box
        box_ls.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))

    features = processor.inference(image_cv, box_ls)
    landmarks_all = np.asarray(features["landmarks"], dtype=np.float32)
    if landmarks_all.size == 0:
        return None

    # Pick the largest detected face.
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    idx = int(np.argmax(areas))
    return landmarks_all[idx]


def compute_nme(source_ldm: np.ndarray, output_ldm: np.ndarray):
    d_interocular = float(np.linalg.norm(source_ldm[36] - source_ldm[45]))
    if d_interocular < 1e-6:
        return None

    point_dist = np.linalg.norm(output_ldm - source_ldm, axis=1)
    result = {"nme_full": float(np.mean(point_dist) / d_interocular)}

    for region_name, idxs in LOCAL_REGIONS.items():
        region_dist = point_dist[idxs]
        result[f"nme_{region_name}"] = float(np.mean(region_dist) / d_interocular)

    return result


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPIGA landmark NME for baseline/ours/stable-makeup.")
    parser.add_argument(
        "--project_root",
        default="/DATA/yantongliu/PSGAN-ours/PSGAN-spiga",
        help="Root of PSGAN-spiga project.",
    )
    parser.add_argument(
        "--baseline_dir",
        default="results/baseline_updated30",
        help="Baseline results dir relative to project_root or absolute path.",
    )
    parser.add_argument(
        "--ours_dir",
        default="results/ours_updated30",
        help="Ours results dir relative to project_root or absolute path.",
    )
    parser.add_argument(
        "--stable_dir",
        default="results/stablemakeup_updated30",
        help="Stable-Makeup results dir relative to project_root or absolute path.",
    )
    parser.add_argument(
        "--pairs_file",
        default="assets/images/pairs.txt",
        help="pairs.txt path relative to project_root or absolute path.",
    )
    parser.add_argument(
        "--source_dir",
        default="assets/images/non-makeup",
        help="source image dir relative to project_root or absolute path.",
    )
    parser.add_argument(
        "--out_prefix",
        default="results/nme_comparison",
        help="Output prefix relative to project_root or absolute path.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    def resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (project_root / p)

    baseline_dir = resolve_path(args.baseline_dir)
    ours_dir = resolve_path(args.ours_dir)
    stable_dir = resolve_path(args.stable_dir)
    pairs_file = resolve_path(args.pairs_file)
    source_dir = resolve_path(args.source_dir)
    out_prefix = resolve_path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(pairs_file)
    methods = ["baseline", "ours", "stablemakeup"]
    method_metric_values = {
        m: {k: [] for k in ["nme_full", "nme_jaw", "nme_brow", "nme_nose", "nme_eyes", "nme_mouth"]}
        for m in methods
    }

    processor = SPIGAFramework(ModelConfig("300wpublic"))
    detector = FaceDetector()

    per_pair_rows = []
    source_fail_count = 0

    for pair_id, (src_name, _) in enumerate(pairs, start=1):
        src_path = source_dir / src_name
        if not src_path.exists():
            source_fail_count += 1
            continue

        src_ldm = detect_landmarks_68(src_path, detector=detector, processor=processor)
        if src_ldm is None:
            source_fail_count += 1
            continue

        method_paths = {
            "baseline": baseline_dir / f"pair{pair_id:02d}_result.png",
            "ours": ours_dir / f"pair{pair_id:02d}_result.png",
            "stablemakeup": resolve_stable_path(stable_dir, pair_id),
        }

        for method in methods:
            out_path = method_paths[method]
            row = {"pair_id": pair_id, "method": method, "image_path": str(out_path) if out_path else ""}

            if out_path is None or (not out_path.exists()):
                row["status"] = "missing_output"
                per_pair_rows.append(row)
                continue

            try:
                out_ldm = detect_landmarks_68(out_path, detector=detector, processor=processor)
            except Exception as e:
                row["status"] = f"error:{e}"
                per_pair_rows.append(row)
                continue

            if out_ldm is None:
                row["status"] = "output_face_not_found"
                per_pair_rows.append(row)
                continue

            metrics = compute_nme(src_ldm, out_ldm)
            if metrics is None:
                row["status"] = "invalid_interocular"
                per_pair_rows.append(row)
                continue

            row["status"] = "ok"
            row.update(metrics)
            per_pair_rows.append(row)

            for k, v in metrics.items():
                method_metric_values[method][k].append(v)

    per_pair_csv = out_prefix.with_name(out_prefix.name + "_per_pair.csv")
    with per_pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "method",
                "status",
                "image_path",
                "nme_full",
                "nme_jaw",
                "nme_brow",
                "nme_nose",
                "nme_eyes",
                "nme_mouth",
            ],
        )
        writer.writeheader()
        for row in per_pair_rows:
            writer.writerow(row)

    summary_rows = []
    for method in methods:
        row = {"method": method}
        for metric_name, values in method_metric_values[method].items():
            if len(values) == 0:
                row[f"{metric_name}_mean"] = ""
                row[f"{metric_name}_std"] = ""
                row[f"{metric_name}_n"] = 0
            else:
                mean, std, n = summarize_metric(values)
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_std"] = std
                row[f"{metric_name}_n"] = n
        summary_rows.append(row)

    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary_fields = ["method"]
    for m in ["nme_full", "nme_jaw", "nme_brow", "nme_nose", "nme_eyes", "nme_mouth"]:
        summary_fields += [f"{m}_mean", f"{m}_std", f"{m}_n"]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Source landmark fail count: {source_fail_count}")
    print(f"Saved per-pair metrics: {per_pair_csv}")
    print(f"Saved summary: {summary_csv}")
    print("\n=== NME Full (lower is better) ===")
    for row in summary_rows:
        print(
            f"{row['method']:12s} mean={row['nme_full_mean'] if row['nme_full_mean'] != '' else 'NA'} "
            f"std={row['nme_full_std'] if row['nme_full_std'] != '' else 'NA'} "
            f"n={row['nme_full_n']}"
        )


if __name__ == "__main__":
    main()
