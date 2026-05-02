"""
Batch makeup transfer demo for evaluation.
Reads source-reference pairs from pairs.txt and runs each through the model.

Usage:
    python3 batch_demo.py --model_path <path_to_G.pth> --output_dir <output_folder>
"""
import argparse
import os
from pathlib import Path
from PIL import Image
from psgan import Inference, PostProcess
from setup import setup_config, setup_argparser


def load_pairs(pairs_file):
    """Read pairs.txt; return list of (source, reference) filename tuples."""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"  [WARN] Skipping malformed line: {line}")
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = setup_argparser()
    parser.add_argument("--model_path", required=True, help="Path to G.pth")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--source_dir", default="assets/images/non-makeup")
    parser.add_argument("--reference_dir", default="assets/images/makeup")
    parser.add_argument("--pairs_file", default="assets/images/pairs.txt")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save results (will be created)")
    args = parser.parse_args()

    config = setup_config(args)

    # Initialize once (model loaded once, used for all pairs)
    print(f"Loading model from {args.model_path}...")
    inference = Inference(config, args.device, args.model_path)
    postprocess = PostProcess(config)

    # Load pairs
    pairs = load_pairs(args.pairs_file)
    print(f"Loaded {len(pairs)} pairs from {args.pairs_file}")

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, (src_name, ref_name) in enumerate(pairs, start=1):
        pair_id = f"pair{i:02d}"
        src_path = Path(args.source_dir) / src_name
        ref_path = Path(args.reference_dir) / ref_name

        if not src_path.exists():
            print(f"  [SKIP] {pair_id}: source not found ({src_path})")
            fail_count += 1
            continue
        if not ref_path.exists():
            print(f"  [SKIP] {pair_id}: reference not found ({ref_path})")
            fail_count += 1
            continue

        print(f"  Running {pair_id}: {src_name} + {ref_name}")
        source = Image.open(src_path).convert("RGB")
        reference = Image.open(ref_path).convert("RGB")

        try:
            image, face = inference.transfer(source, reference, with_face=True)
            if image is None:
                print(f"  [FAIL] {pair_id}: face detection failed")
                fail_count += 1
                continue

            source_crop = source.crop(
                (face.left(), face.top(), face.right(), face.bottom()))
            image = postprocess(source_crop, image)

            output_path = Path(args.output_dir) / f"{pair_id}_result.png"
            image.save(output_path)
            print(f"  [OK]   {pair_id} -> {output_path}")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {pair_id}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\nDone! {success_count} succeeded, {fail_count} failed")
    print(f"Results in {args.output_dir}")


if __name__ == '__main__':
    main()
