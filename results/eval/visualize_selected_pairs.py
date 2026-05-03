from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAIR_IDS = [2, 4, 8, 19, 23, 28]


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


def find_stable_image(stable_dir: Path, pair_id: int) -> Path:
    prefix = f"pair{pair_id:02d}_"
    matches = sorted(stable_dir.glob(f"{prefix}*.png"))
    if not matches:
        raise FileNotFoundError(f"No stable-makeup output found for {prefix} in {stable_dir}")
    return matches[0]


def load_and_fit(img_path: Path, size: int) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main():
    project_root = Path("/DATA/yantongliu/PSGAN-ours/PSGAN-spiga")
    results_dir = project_root / "results"

    pairs_file = project_root / "assets/images/pairs.txt"
    source_dir = project_root / "assets/images/non-makeup"
    reference_dir = project_root / "assets/images/makeup"
    baseline_dir = results_dir / "baseline_updated30"
    ours_dir = results_dir / "ours_updated30"
    stable_dir = results_dir / "stablemakeup_updated30"

    pairs = parse_pairs(pairs_file)

    columns = [
        "Source-img",
        "PSGAN",
        "ours",
        "Stable-Makeup",
        "Ref-makeup",
    ]

    cell_size = 256
    left_label_w = 0
    top_header_h = 70
    gap = 14
    pad = 20

    rows = len(PAIR_IDS)
    cols = len(columns)

    canvas_w = pad * 2 + left_label_w + cols * cell_size + (cols - 1) * gap
    canvas_h = pad * 2 + top_header_h + rows * cell_size + (rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        header_font = ImageFont.load_default()

    for c, name in enumerate(columns):
        cell_x = pad + left_label_w + c * (cell_size + gap)
        text_bbox = draw.textbbox((0, 0), name, font=header_font)
        text_w = text_bbox[2] - text_bbox[0]
        x = cell_x + max((cell_size - text_w) // 2, 0)
        y = pad + 18
        draw.text((x, y), name, fill="black", font=header_font)

    for r, pair_id in enumerate(PAIR_IDS):
        source_name, ref_name = pairs[pair_id - 1]
        row_y = pad + top_header_h + r * (cell_size + gap)

        images = [
            source_dir / source_name,
            baseline_dir / f"pair{pair_id:02d}_result.png",
            ours_dir / f"pair{pair_id:02d}_result.png",
            find_stable_image(stable_dir, pair_id),
            reference_dir / ref_name,
        ]

        for c, img_path in enumerate(images):
            x = pad + left_label_w + c * (cell_size + gap)
            img = load_and_fit(img_path, cell_size)
            canvas.paste(img, (x, row_y))

    out_path = results_dir / "comparison_pairs_02_04_08_19_23_28.png"
    canvas.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
