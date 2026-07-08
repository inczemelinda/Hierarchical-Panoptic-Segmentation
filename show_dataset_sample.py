"""
Display one sample from each annotation folder of a PhenoBench split in a 3x2 grid.

For a single image name, it shows the corresponding file from:
    images, semantics, plant_instances, plant_visibility,
    leaf_instances, leaf_visibility
Each panel is titled with the folder name (Times New Roman, size 12).

Usage:
    python show_dataset_sample.py
    python show_dataset_sample.py --data-dir /nvmedrive/PhenoBench --split train
    python show_dataset_sample.py --image 05-15_00028_P0030852.png --save sample.png
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Times New Roman for all text (falls back to a serif font if not installed).
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

# Folder order: plants on the middle row, leaves on the bottom row.
FOLDERS = [
    "images",
    "semantics",
    "plant_instances",
    "plant_visibility",
    "leaf_instances",
    "leaf_visibility",
]


def colorize_labels(arr):
    """Map each non-zero id (semantic class or instance) to a distinct color."""
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    palette = plt.colormaps["tab20"].colors
    for i, value in enumerate(np.unique(arr)):
        if value == 0:
            continue  # background stays black
        color = tuple(int(c * 255) for c in palette[i % len(palette)])
        out[arr == value] = color
    return out


def load_panel(folder, path):
    """Return (image_array, imshow_kwargs) prepared for display."""
    arr = np.array(Image.open(path))
    if folder == "images":
        return arr, {}
    if folder.endswith("_visibility"):
        return arr, {"cmap": "gray", "vmin": 0, "vmax": 255}
    # semantics and *_instances -> colorized labels
    return colorize_labels(arr), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/nvmedrive/PhenoBench",
                        help="PhenoBench root directory")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--image", default=None,
                        help="Specific image filename (defaults to the first one)")
    parser.add_argument("--save", default="dataset_sample.png",
                        help="Output path for the figure")
    args = parser.parse_args()

    split_dir = os.path.join(args.data_dir, args.split)
    images_dir = os.path.join(split_dir, "images")

    image_name = args.image
    if image_name is None:
        image_name = sorted(os.listdir(images_dir))[0]
    print(f"Showing sample: {image_name}")

    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    axes = axes.ravel()

    for ax, folder in zip(axes, FOLDERS):
        ax.set_title(folder, fontsize=12)
        ax.axis("off")
        path = os.path.join(split_dir, folder, image_name)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            continue
        panel, kwargs = load_panel(folder, path)
        ax.imshow(panel, **kwargs)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
