"""
Run inference on test images using a trained model.
Saves colored panoptic predictions (plant + leaf instances + overlay).

Usage:
    python run_inference.py --config-file configs/phenobench/exp_full_aug_tversky.yaml \
                            --weights output/Exp_FullAug_Tversky_R50/model_final.pth \
                            --num-images 20
"""
import argparse
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from register_phenobench import register_phenobench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="Path to model config YAML")
    parser.add_argument("--weights", required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--input-dir", default="/nvmedrive/PhenoBench/test/images",
                        help="Directory containing test images")
    parser.add_argument("--output-dir", default="output/predictions",
                        help="Where to save predictions")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of images to process (0 = all)")
    args = parser.parse_args()

    # Build config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()

    # Register dataset (needed for metadata)
    meta = {
        "thing_dataset_id_to_contiguous_id": {1: 1, 2: 2},
        "stuff_dataset_id_to_contiguous_id": {0: 0},
        "thing_classes": ['crop', 'weed'],
        "thing_colors": [(66, 135, 245), (245, 66, 66)],
        "stuff_classes": ['soil']
    }
    register_phenobench("phenobench_test", meta, "/nvmedrive/PhenoBench",
                        split="test", resize_aug=False)

    # Build and load model
    print("Loading model...")
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.weights)

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "plant"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "leaf"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "overlay"), exist_ok=True)

    # Get list of test images
    image_files = sorted(os.listdir(args.input_dir))
    if args.num_images > 0:
        image_files = image_files[:args.num_images]

    print(f"Processing {len(image_files)} images...")

    # Color palette for instance visualization
    colormap = plt.colormaps['Set1'].colors

    for idx, image_name in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] {image_name}")

        # Load and prepare image
        img_path = os.path.join(args.input_dir, image_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]

        # Convert to tensor (C, H, W) format
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32))

        # Run inference
        with torch.no_grad():
            outputs = model([{"image": image_tensor, "image_name": image_name}])

        # Extract panoptic predictions
        plant_panoptic, _ = outputs[0]["plant_panoptic_seg"]
        leaf_panoptic, _ = outputs[0]["leaf_panoptic_seg"]
        plant_panoptic = plant_panoptic.cpu().numpy()
        leaf_panoptic = leaf_panoptic.cpu().numpy()

        # Colorize plant instances
        plant_viz = np.zeros((H, W, 3), dtype=np.uint8)
        for i, plant_id in enumerate(np.unique(plant_panoptic)):
            if plant_id != 0:
                color = tuple(int(c * 255) for c in colormap[i % 9])
                plant_viz[plant_panoptic == plant_id] = color

        # Colorize leaf instances
        leaf_viz = np.zeros((H, W, 3), dtype=np.uint8)
        for i, leaf_id in enumerate(np.unique(leaf_panoptic)):
            if leaf_id != 0:
                color = tuple(int(c * 255) for c in colormap[i % 9])
                leaf_viz[leaf_panoptic == leaf_id] = color

        # Create overlay on original image (50% blend)
        overlay = (0.5 * image + 0.5 * plant_viz).astype(np.uint8)

        # Save all three visualizations
        Image.fromarray(plant_viz).save(os.path.join(args.output_dir, "plant", image_name))
        Image.fromarray(leaf_viz).save(os.path.join(args.output_dir, "leaf", image_name))
        Image.fromarray(overlay).save(os.path.join(args.output_dir, "overlay", image_name))

    print(f"\nDone! Predictions saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
