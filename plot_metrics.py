"""
Generate publication-quality plots from training metrics.json.
Saves PNG files to <output_dir>/plots/.

Usage:
    python plot_metrics.py --output-dir output/Exp_FullAug_Tversky_R50
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_file):
    """Load all metrics entries from JSON lines file."""
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Training output dir (e.g., output/Exp_FullAug_Tversky_R50)")
    args = parser.parse_args()

    metrics_file = os.path.join(args.output_dir, "metrics.json")
    metrics = load_metrics(metrics_file)

    # Filter training metrics (have total_loss)
    train_metrics = [m for m in metrics if "total_loss" in m]

    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ===== PLOT 1: Total loss curve =====
    plt.figure(figsize=(10, 6))
    plt.plot([m["iteration"] for m in train_metrics],
             [m["total_loss"] for m in train_metrics], color='#2E75B6', linewidth=1.5)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_total_loss.png"), dpi=150)
    plt.close()

    # ===== PLOT 2: Plant-level losses =====
    plt.figure(figsize=(12, 6))
    plant_losses = ["plant_loss_ce", "plant_loss_dice", "plant_loss_mask",
                    "plant_loss_boundary", "plant_loss_tversky"]
    for loss_name in plant_losses:
        values = [(m["iteration"], m[loss_name])
                  for m in train_metrics if loss_name in m]
        if values:
            x, y = zip(*values)
            plt.plot(x, y, label=loss_name, alpha=0.8, linewidth=1.2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.title("Plant-Level Loss Components", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_plant_losses.png"), dpi=150)
    plt.close()

    # ===== PLOT 3: Leaf-level losses =====
    plt.figure(figsize=(12, 6))
    leaf_losses = ["leaf_loss_ce", "leaf_loss_dice", "leaf_loss_mask",
                   "leaf_loss_boundary", "leaf_loss_tversky"]
    for loss_name in leaf_losses:
        values = [(m["iteration"], m[loss_name])
                  for m in train_metrics if loss_name in m]
        if values:
            x, y = zip(*values)
            plt.plot(x, y, label=loss_name, alpha=0.8, linewidth=1.2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.title("Leaf-Level Loss Components", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_leaf_losses.png"), dpi=150)
    plt.close()

    # ===== PLOT 4: Learning rate schedule =====
    plt.figure(figsize=(10, 6))
    plt.plot([m["iteration"] for m in train_metrics],
             [m["lr"] for m in train_metrics], color='#C00000', linewidth=1.5)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule (WarmupPolyLR)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_learning_rate.png"), dpi=150)
    plt.close()

    # ===== PLOT 5: Validation metrics =====
    # Detect eval metric keys (Detectron2 may prefix with dataset name like "phenobench_val/")
    eval_iters, pq_values, pq_dagger = [], [], []
    iou_crop, iou_weed, iou_soil = [], [], []

    def get_eval_value(m, key):
        # Try with and without dataset prefix
        for prefix in ["", "phenobench_val/"]:
            full_key = prefix + key
            if full_key in m:
                return m[full_key]
        return None

    for m in metrics:
        pq = get_eval_value(m, "PQ")
        if pq is not None:
            eval_iters.append(m.get("iteration", 0))
            pq_values.append(pq)
            pq_dagger.append(get_eval_value(m, "PQ+") or 0)
            iou_crop.append(get_eval_value(m, "IoU (crop)") or 0)
            iou_weed.append(get_eval_value(m, "IoU (weed)") or 0)
            iou_soil.append(get_eval_value(m, "IoU (soil)") or 0)

    if eval_iters:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(eval_iters, pq_values, marker='o', label='PQ', color='#1F4E79')
        axes[0].plot(eval_iters, pq_dagger, marker='s', label='PQ+', color='#1A6B3C')
        axes[0].set_xlabel("Iteration", fontsize=12)
        axes[0].set_ylabel("Panoptic Quality", fontsize=12)
        axes[0].set_title("PQ Metrics During Training", fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(eval_iters, iou_crop, marker='o', label='IoU (crop)', color='#2E75B6')
        axes[1].plot(eval_iters, iou_weed, marker='s', label='IoU (weed)', color='#C00000')
        axes[1].plot(eval_iters, iou_soil, marker='^', label='IoU (soil)', color='#806000')
        axes[1].set_xlabel("Iteration", fontsize=12)
        axes[1].set_ylabel("IoU", fontsize=12)
        axes[1].set_title("Per-Class IoU During Training", fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "05_eval_metrics.png"), dpi=150)
        plt.close()

    print(f"Plots saved to {plots_dir}/")
    print("Files generated:")
    for f in sorted(os.listdir(plots_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
