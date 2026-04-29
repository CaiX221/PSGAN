"""
Generate SPIGA 68-point structure maps for PSGAN training data.
Based on spiga_draw.py from Stable-Makeup project.

Usage: python generate_spiga.py
Output: data/spiga/makeup/ and data/spiga/non-makeup/
"""

import numpy as np
import tqdm
from PIL import Image
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


# Initialize SPIGA processor and face detector
processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector()


def get_landmarks(image, detector):
    """Detect face and extract 68 landmarks using SPIGA."""
    image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces, boxes, scores, landmarks = detector.detect_align(image_cv)
    boxes = boxes.cpu().numpy()
    box_ls = []
    for box in boxes:
        x, y, x1, y1 = box
        box = x, y, x1 - x, y1 - y
        box_ls.append(box)
    if len(box_ls) == 0:
        return []
    features = processor.inference(image_cv, box_ls)
    return np.array(features['landmarks'])


def get_patch(landmarks, color='lime', closed=False):
    """Create a matplotlib patch from landmark points."""
    contour = landmarks
    ops = [Path.MOVETO] + [Path.LINETO] * (len(contour) - 1)
    facecolor = (0, 0, 0, 0)
    if closed:
        contour.append(contour[0])
        ops.append(Path.CLOSEPOLY)
        facecolor = color
    path = Path(contour, ops)
    return patches.PathPatch(path, facecolor=facecolor, edgecolor=color, lw=4)


def draw_structure_map(landmarks_list, size=361):
    """Draw colored 68-point landmark map on black background.

    Colors match Stable-Makeup's convention:
        Green = jawline, Yellow = eyebrows, Orange = nose,
        Magenta = eyes, Cyan = outer lips, Blue = inner lips
    """
    dpi = 72
    fig, ax = plt.subplots(1, figsize=[size / dpi, size / dpi], tight_layout={'pad': 0})
    fig.set_dpi(dpi)

    black = np.zeros((size, size, 3))
    ax.imshow(black)

    for landmarks in landmarks_list:
        ax.add_patch(get_patch(landmarks[0:17]))                                  # jawline - green
        ax.add_patch(get_patch(landmarks[17:22], color='yellow'))                  # left eyebrow
        ax.add_patch(get_patch(landmarks[22:27], color='yellow'))                  # right eyebrow
        ax.add_patch(get_patch(landmarks[27:31], color='orange'))                  # nose vertical
        ax.add_patch(get_patch(landmarks[31:36], color='orange'))                  # nose horizontal
        ax.add_patch(get_patch(landmarks[36:42], color='magenta', closed=True))    # left eye
        ax.add_patch(get_patch(landmarks[42:48], color='magenta', closed=True))    # right eye
        ax.add_patch(get_patch(landmarks[48:60], color='cyan', closed=True))       # outer lips
        ax.add_patch(get_patch(landmarks[60:68], color='blue', closed=True))       # inner lips

    plt.axis('off')
    fig.canvas.draw()
    buffer, (width, height) = fig.canvas.print_to_buffer()
    buffer = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    buffer = buffer[:, :, 0:3]  # drop alpha channel
    plt.close(fig)
    return Image.fromarray(buffer).resize((size, size))


def parse_landmarks(landmarks):
    """Convert landmark array to list of (x, y) tuples."""
    ldm = []
    for landmark in landmarks:
        ldm.append([(float(x), float(y)) for x, y in landmark])
    return ldm


def process_folder(input_dir, output_dir, size=361):
    """Generate SPIGA structure maps for all images in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    image_names = sorted(os.listdir(input_dir))
    success = 0
    failed = 0

    pbar = tqdm.tqdm(image_names, desc=f"Processing {input_dir}")
    for name in pbar:
        output_path = os.path.join(output_dir, name)
        # Skip if already generated
        if os.path.exists(output_path):
            success += 1
            continue

        try:
            image = Image.open(os.path.join(input_dir, name)).convert("RGB")
            landmarks = get_landmarks(image, detector)

            if len(landmarks) == 0:
                failed += 1
                pbar.set_postfix(success=success, failed=failed)
                continue

            # Parse landmarks and draw structure map
            parsed = parse_landmarks(landmarks)
            structure_map = draw_structure_map(parsed, size=size)
            structure_map.save(output_path)
            success += 1

        except Exception as e:
            print(f"Error processing {name}: {e}")
            failed += 1

        pbar.set_postfix(success=success, failed=failed)

    print(f"\nDone: {success} succeeded, {failed} failed out of {len(image_names)}")


if __name__ == '__main__':
    print("Generating SPIGA structure maps for PSGAN training data...")
    print("Image size: 361x361")
    print()

    # Process makeup images
    print("=== Processing makeup images ===")
    process_folder("data/images/makeup/", "data/spiga/makeup/", size=361)

    # Process non-makeup images
    print("\n=== Processing non-makeup images ===")
    process_folder("data/images/non-makeup/", "data/spiga/non-makeup/", size=361)

    print("\nAll done! Structure maps saved to data/spiga/")