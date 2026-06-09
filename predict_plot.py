"""
Run prediction on a single image and show a clean side-by-side strip of the
5 visualizations (input, plant panoptic, plant overlay, leaf panoptic,
leaf overlay) — same content as the webapp, but as one wide image with no
titles, axes or gaps between panels.

Usage:
    python predict_plot.py --image /path/to/img.png
    python predict_plot.py --image /path/to/img.png --model SwinL
    python predict_plot.py --image /path/to/img.png --save strip.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Same registry shape as webapp/app.py.
MODELS = {
    "R50": {
        "config": os.path.join(PROJECT_ROOT, "configs/phenobench/exp_full_aug_tversky.yaml"),
        "weights": os.path.join(PROJECT_ROOT, "output/Exp_FullAug_Tversky_R50/model_final.pth"),
    },
    "SwinL": {
        "config": os.path.join(PROJECT_ROOT, "configs/phenobench/exp_swinL_tversky.yaml"),
        "weights": os.path.join(PROJECT_ROOT, "output/Exp_SwinL_Tversky/model_final.pth"),
    },
}


def _register_phenobench_metadata():
    """Minimal metadata so the model's from_config does not complain."""
    meta = {
        "thing_dataset_id_to_contiguous_id": {1: 1, 2: 2},
        "stuff_dataset_id_to_contiguous_id": {0: 0},
        "thing_classes": ["crop", "weed"],
        "thing_colors": [(66, 135, 245), (245, 66, 66)],
        "stuff_classes": ["soil"],
    }
    for name in ("phenobench_train", "phenobench_val", "phenobench_test"):
        md = MetadataCatalog.get(name)
        if "thing_classes" in md.as_dict():
            continue
        md.set(
            evaluator_type="phenobench",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


def load_model(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")
    info = MODELS[model_name]
    if not os.path.exists(info["weights"]):
        raise FileNotFoundError(f"Weights not found: {info['weights']}")

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(info["config"])
    cfg.MODEL.WEIGHTS = info["weights"]
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    _register_phenobench_metadata()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(info["weights"])
    return model


def predict(model, image_path):
    pil = Image.open(image_path).convert("RGB")
    image = np.array(pil)
    H, W = image.shape[:2]
    tensor = torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32))

    with torch.no_grad():
        outputs = model([{
            "image": tensor,
            "image_name": os.path.basename(image_path),
            "height": H,
            "width": W,
        }])

    plant_pan, _ = outputs[0]["plant_panoptic_seg"]
    leaf_pan, _ = outputs[0]["leaf_panoptic_seg"]
    plant_pan = plant_pan.cpu().numpy()
    leaf_pan = leaf_pan.cpu().numpy()

    colormap = plt.colormaps["Set1"].colors

    plant_viz = np.zeros((H, W, 3), dtype=np.uint8)
    for i, pid in enumerate(np.unique(plant_pan)):
        if pid != 0:
            color = tuple(int(c * 255) for c in colormap[i % 9])
            plant_viz[plant_pan == pid] = color

    leaf_viz = np.zeros((H, W, 3), dtype=np.uint8)
    for i, lid in enumerate(np.unique(leaf_pan)):
        if lid != 0:
            color = tuple(int(c * 255) for c in colormap[i % 9])
            leaf_viz[leaf_pan == lid] = color

    plant_overlay = (0.5 * image + 0.5 * plant_viz).astype(np.uint8)
    leaf_overlay = (0.5 * image + 0.5 * leaf_viz).astype(np.uint8)

    # Order matches the webapp: input, plant panoptic, plant overlay, leaf panoptic, leaf overlay
    return [image, plant_viz, plant_overlay, leaf_viz, leaf_overlay]


def show_strip(images, save_path=None):
    """Concatenate horizontally with zero gaps and display (and optionally save)."""
    strip = np.hstack(images)
    H, W = strip.shape[:2]
    # 4-inch tall figure, width scaled to preserve aspect ratio.
    fig_h = 4.0
    fig_w = fig_h * (W / H)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])  # axes fill the whole figure — no padding
    ax.imshow(strip)
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches=None, pad_inches=0)
        print(f"Saved strip to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", default="R50", choices=list(MODELS.keys()),
                        help="Which trained model to use (default: R50)")
    parser.add_argument("--save", default=None,
                        help="Optional output path for the combined strip PNG")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    model = load_model(args.model)
    print(f"Running inference on {args.image}...")
    images = predict(model, args.image)
    show_strip(images, save_path=args.save)


if __name__ == "__main__":
    main()
