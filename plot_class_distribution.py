"""
Plot the class distribution of the PhenoBench dataset to show class imbalance.

It scans the ground-truth annotations of a split (or of train+val together) and
produces two views of the imbalance:

  (A) Semantic distribution at the PIXEL level  -> soil vs crop vs weed
      (soil dominates by a large margin; weed is a tiny fraction of the pixels)

  (B) Number of annotated INSTANCES              -> crop vs weed vs leaf
      (how many objects of each kind are labelled)

Semantic labels follow the PhenoBench convention:
    0 = soil, 1 = crop, 2 = weed, 3 = partial-crop, 4 = partial-weed, 255 = ignore
Partial plants are remapped to their full class (3 -> crop, 4 -> weed), exactly
as during training.

Usage (run from WSL2, where the dataset lives):

    python plot_class_distribution.py --phenobench_dir /nvmedrive/PhenoBench --split train
    python plot_class_distribution.py --phenobench_dir /nvmedrive/PhenoBench --split all --out class_distribution.png
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# Times New Roman 12 pt where available, serif fallback otherwise (e.g. on Linux).
# No fixed backend is selected, so the figure opens in a window when a display
# is available (e.g. WSLg) and is always saved to disk as well.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(x, **kwargs):
        return x

CLASS_NAMES = {0: "soil", 1: "crop", 2: "weed"}
SEM_COLORS = {"soil": "#8d6e63", "crop": "#4a7c59", "weed": "#c0392b"}
INST_COLORS = {"crop": "#4a7c59", "weed": "#c0392b", "leaf": "#2e6b39"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phenobench_dir", default="/nvmedrive/PhenoBench", type=Path,
                   help="Root of the PhenoBench dataset.")
    p.add_argument("--split", default="train", type=str,
                   choices=["train", "val", "all"],
                   help="Split to analyse ('all' = train + val).")
    p.add_argument("--out", default="class_distribution.png", type=str,
                   help="Output image path.")
    return p.parse_args()


def count_split(root: Path, split: str, acc):
    """Accumulate pixel and instance counts for one split into `acc`."""
    sem_dir = root / split / "semantics"
    plant_dir = root / split / "plant_instances"
    leaf_dir = root / split / "leaf_instances"

    names = sorted(os.listdir(sem_dir))
    for name in tqdm(names, desc=f"Scanning {split}"):
        sem = np.array(Image.open(sem_dir / name))
        # Remap partial -> full class, matching the training pipeline.
        sem = sem.copy()
        sem[sem == 3] = 1
        sem[sem == 4] = 2

        # (A) pixels per semantic class
        for label in (0, 1, 2):
            acc["pixels"][label] += int(np.count_nonzero(sem == label))

        # (B) instance counts
        plant = np.array(Image.open(plant_dir / name))
        for label, key in ((1, "crop"), (2, "weed")):
            ids = np.unique(plant[sem == label])
            ids = ids[ids != 0]
            acc["instances"][key] += int(len(ids))

        leaf = np.array(Image.open(leaf_dir / name))
        leaf_ids = np.unique(leaf[leaf > 0])
        acc["instances"]["leaf"] += int(len(leaf_ids))

        acc["n_images"] += 1


def main():
    args = parse_args()
    assert args.phenobench_dir.exists(), \
        f"Dataset not found at {args.phenobench_dir}"

    splits = ["train", "val"] if args.split == "all" else [args.split]
    acc = {
        "pixels": {0: 0, 1: 0, 2: 0},
        "instances": {"crop": 0, "weed": 0, "leaf": 0},
        "n_images": 0,
    }
    for split in splits:
        count_split(args.phenobench_dir, split, acc)

    # ---- console summary ----
    total_px = sum(acc["pixels"].values())
    print("\n================ Class distribution "
          f"({' + '.join(splits)}, {acc['n_images']} images) ================")
    print("\nSemantic pixels:")
    for label in (0, 1, 2):
        px = acc["pixels"][label]
        print(f"  {CLASS_NAMES[label]:>5}: {px:>14,d} px  ({100 * px / total_px:6.2f} %)")
    px_crop, px_weed = acc["pixels"][1], acc["pixels"][2]
    print(f"  -> soil/weed pixel ratio : {acc['pixels'][0] / px_weed:,.0f} : 1")
    print(f"  -> crop/weed pixel ratio : {px_crop / px_weed:,.1f} : 1")

    print("\nInstances:")
    for key in ("crop", "weed", "leaf"):
        print(f"  {key:>5}: {acc['instances'][key]:>8,d}")
    print(f"  -> crop/weed instance ratio: "
          f"{acc['instances']['crop'] / acc['instances']['weed']:.2f} : 1")

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (A) semantic pixels, log scale (imbalance spans orders of magnitude)
    names = [CLASS_NAMES[i] for i in (0, 1, 2)]
    vals = [acc["pixels"][i] for i in (0, 1, 2)]
    bars = axA.bar(names, vals, color=[SEM_COLORS[n] for n in names],
                   edgecolor="black", linewidth=0.6, width=0.6)
    axA.set_yscale("log")
    axA.set_ylabel("Number of pixels (log scale)")
    axA.set_title("(a) Semantic class distribution (pixels)")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v * 1.15,
                 f"{100 * v / total_px:.2f}%", ha="center", va="bottom", fontsize=12)
    axA.set_ylim(top=max(vals) * 4)
    axA.margins(x=0.15)

    # (B) instance counts
    ikeys = ["crop", "weed", "leaf"]
    ivals = [acc["instances"][k] for k in ikeys]
    bars2 = axB.bar(ikeys, ivals, color=[INST_COLORS[k] for k in ikeys],
                    edgecolor="black", linewidth=0.6, width=0.6)
    axB.set_ylabel("Number of annotated instances")
    axB.set_title("(b) Instance count per class")
    for b, v in zip(bars2, ivals):
        axB.text(b.get_x() + b.get_width() / 2, v + max(ivals) * 0.01,
                 f"{v:,}", ha="center", va="bottom", fontsize=12)
    axB.margins(y=0.15)

    fig.suptitle("PhenoBench class distribution "
                 f"({' + '.join(splits)} split, {acc['n_images']} images)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to: {os.path.abspath(args.out)}")

    # Open the figure in a window when a display is available (WSLg / desktop).
    plt.show()


if __name__ == "__main__":
    main()
