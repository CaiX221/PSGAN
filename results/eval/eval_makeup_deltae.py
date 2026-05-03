import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab


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


def get_eye_mask_from_landmarks(landmarks_yx: np.ndarray, size: int = 512):
    mask = np.zeros((size, size), dtype=np.uint8)
    left_eye_xy = np.stack([landmarks_yx[36:42, 1], landmarks_yx[36:42, 0]], axis=1).astype(np.int32)
    right_eye_xy = np.stack([landmarks_yx[42:48, 1], landmarks_yx[42:48, 0]], axis=1).astype(np.int32)
    cv2.fillConvexPoly(mask, left_eye_xy, 1)
    cv2.fillConvexPoly(mask, right_eye_xy, 1)
    return mask.astype(bool)


def mean_lab_of_region(lab_img: np.ndarray, region_mask: np.ndarray):
    if region_mask.sum() < 20:
        return None
    return lab_img[region_mask].mean(axis=0)


def compute_deltae00(mean_lab_a: np.ndarray, mean_lab_b: np.ndarray):
    a = mean_lab_a.reshape(1, 1, 3)
    b = mean_lab_b.reshape(1, 1, 3)
    return float(deltaE_ciede2000(a, b)[0, 0])


def extract_region_means(
    image_path: Path,
    parser,
    dlib_detect,
    dlib_landmarks,
):
    img = Image.open(image_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    img_np = np.asarray(img)

    faces = dlib_detect(img)
    if len(faces) == 0:
        return None, "face_not_found"

    lms = dlib_landmarks(img, faces[0])  # (68, 2), (y, x)
    eye_mask = get_eye_mask_from_landmarks(lms, size=512)

    parse_map = parser.parse(img_np).squeeze().cpu().numpy().astype(np.int32)
    lip_mask = (parse_map == 7) | (parse_map == 9)
    skin_mask = ((parse_map == 1) | (parse_map == 6)) & (~lip_mask) & (~eye_mask)

    rgb = img_np.astype(np.float32) / 255.0
    lab = rgb2lab(rgb)

    means = {
        "lip": mean_lab_of_region(lab, lip_mask),
        "eye": mean_lab_of_region(lab, eye_mask),
        "skin": mean_lab_of_region(lab, skin_mask),
    }
    if any(v is None for v in means.values()):
        return None, "region_too_small"
    return means, "ok"


def summarize(values):
    if len(values) == 0:
        return "", "", 0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def main():
    parser = argparse.ArgumentParser(description="Evaluate makeup transfer by regional CIEDE2000 (DeltaE00).")
    parser.add_argument("--project_root", default="/DATA/yantongliu/PSGAN-ours/PSGAN-spiga")
    parser.add_argument("--pairs_file", default="assets/images/pairs.txt")
    parser.add_argument("--reference_dir", default="assets/images/makeup")
    parser.add_argument("--baseline_dir", default="results/baseline_updated30")
    parser.add_argument("--ours_dir", default="results/ours_updated30")
    parser.add_argument("--stable_dir", default="results/stablemakeup_updated30")
    parser.add_argument("--output_prefix", default="results/output/deltae_regional")
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
    dlib_detect = futils.dlib.detect
    dlib_landmarks = futils.dlib.landmarks

    methods = ["baseline", "ours", "stablemakeup"]
    metric_store = {m: {"deltae_lip": [], "deltae_eye": [], "deltae_skin": [], "deltae_avg": []} for m in methods}
    rows = []

    ref_cache = {}

    for pair_id, (_, ref_name) in enumerate(pairs, start=1):
        ref_path = reference_dir / ref_name
        if not ref_path.exists():
            for m in methods:
                rows.append(
                    {
                        "pair_id": pair_id,
                        "method": m,
                        "status": "missing_reference",
                        "reference_path": str(ref_path),
                        "output_path": "",
                    }
                )
            continue

        if pair_id not in ref_cache:
            ref_means, ref_status = extract_region_means(ref_path, face_parser, dlib_detect, dlib_landmarks)
            ref_cache[pair_id] = (ref_means, ref_status)

        ref_means, ref_status = ref_cache[pair_id]
        if ref_status != "ok":
            for m in methods:
                rows.append(
                    {
                        "pair_id": pair_id,
                        "method": m,
                        "status": f"reference_{ref_status}",
                        "reference_path": str(ref_path),
                        "output_path": "",
                    }
                )
            continue

        outputs = {
            "baseline": baseline_dir / f"pair{pair_id:02d}_result.png",
            "ours": ours_dir / f"pair{pair_id:02d}_result.png",
            "stablemakeup": resolve_stable_output(stable_dir, pair_id),
        }

        for method in methods:
            out_path = outputs[method]
            row = {
                "pair_id": pair_id,
                "method": method,
                "reference_path": str(ref_path),
                "output_path": str(out_path) if out_path else "",
            }

            if out_path is None or not out_path.exists():
                row["status"] = "missing_output"
                rows.append(row)
                continue

            out_means, out_status = extract_region_means(out_path, face_parser, dlib_detect, dlib_landmarks)
            if out_status != "ok":
                row["status"] = f"output_{out_status}"
                rows.append(row)
                continue

            d_lip = compute_deltae00(out_means["lip"], ref_means["lip"])
            d_eye = compute_deltae00(out_means["eye"], ref_means["eye"])
            d_skin = compute_deltae00(out_means["skin"], ref_means["skin"])
            d_avg = float(np.mean([d_lip, d_eye, d_skin]))

            row.update(
                {
                    "status": "ok",
                    "deltae_lip": d_lip,
                    "deltae_eye": d_eye,
                    "deltae_skin": d_skin,
                    "deltae_avg": d_avg,
                }
            )
            rows.append(row)

            metric_store[method]["deltae_lip"].append(d_lip)
            metric_store[method]["deltae_eye"].append(d_eye)
            metric_store[method]["deltae_skin"].append(d_skin)
            metric_store[method]["deltae_avg"].append(d_avg)

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
                "deltae_lip",
                "deltae_eye",
                "deltae_skin",
                "deltae_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in methods:
        lip_mean, lip_std, lip_n = summarize(metric_store[method]["deltae_lip"])
        eye_mean, eye_std, eye_n = summarize(metric_store[method]["deltae_eye"])
        skin_mean, skin_std, skin_n = summarize(metric_store[method]["deltae_skin"])
        avg_mean, avg_std, avg_n = summarize(metric_store[method]["deltae_avg"])
        summary_rows.append(
            {
                "method": method,
                "deltae_lip_mean": lip_mean,
                "deltae_lip_std": lip_std,
                "deltae_lip_n": lip_n,
                "deltae_eye_mean": eye_mean,
                "deltae_eye_std": eye_std,
                "deltae_eye_n": eye_n,
                "deltae_skin_mean": skin_mean,
                "deltae_skin_std": skin_std,
                "deltae_skin_n": skin_n,
                "deltae_avg_mean": avg_mean,
                "deltae_avg_std": avg_std,
                "deltae_avg_n": avg_n,
            }
        )

    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "deltae_lip_mean",
                "deltae_lip_std",
                "deltae_lip_n",
                "deltae_eye_mean",
                "deltae_eye_std",
                "deltae_eye_n",
                "deltae_skin_mean",
                "deltae_skin_std",
                "deltae_skin_n",
                "deltae_avg_mean",
                "deltae_avg_std",
                "deltae_avg_n",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-pair metrics: {per_pair_csv}")
    print(f"Saved summary: {summary_csv}")
    print("\n=== DeltaE00 (lower is better) ===")
    for r in summary_rows:
        print(
            f"{r['method']:12s} "
            f"lip={r['deltae_lip_mean']} eye={r['deltae_eye_mean']} skin={r['deltae_skin_mean']} avg={r['deltae_avg_mean']}"
        )


if __name__ == "__main__":
    main()
