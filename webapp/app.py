"""
Minimal Flask app for visualizing Hierarchical Mask2Former predictions.

Upload an image, pick a trained model from the dropdown, and the app shows the
original image alongside the plant-level panoptic prediction, the leaf-level
panoptic prediction and a plant-level overlay on the original. Below the
images, the latest evaluation metrics of the selected model are shown
(read from <output_dir>/metrics.json).

Run from the project root:

    python -m webapp.app

The default address is http://localhost:5000.
"""
import base64
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image

# Make the project root importable so this can be launched as `python -m webapp.app`
# or `python webapp/app.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config


app = Flask(__name__)

# registry of the trained models exposed in the user interface
MODELS = {
    "SwinL_baseline": {
        "label": "Swin-L (baseline)",
        "config": os.path.join(PROJECT_ROOT, "configs/phenobench/exp_swinL_baseline.yaml"),
        "weights": os.path.join(PROJECT_ROOT, "output/Exp_SwinL_Baseline/model_final.pth"),
        "output_dir": os.path.join(PROJECT_ROOT, "output/Exp_SwinL_Baseline"),
    },
    "SwinL": {
        "label": "Swin-L (proposed)",
        "config": os.path.join(PROJECT_ROOT, "configs/phenobench/exp_swinL_tversky.yaml"),
        "weights": os.path.join(PROJECT_ROOT, "output/Exp_SwinL_Tversky/model_final.pth"),
        "output_dir": os.path.join(PROJECT_ROOT, "output/Exp_SwinL_Tversky"),
    },
    "Hiera": {
        "label": "Hiera-L",
        "config": os.path.join(PROJECT_ROOT, "configs/phenobench/exp_hiera_tversky.yaml"),
        "weights": os.path.join(PROJECT_ROOT, "output/Exp_Hiera_Tversky/model_final.pth"),
        "output_dir": os.path.join(PROJECT_ROOT, "output/Exp_Hiera_Tversky"),
    },
}

# Order in which metric cards are rendered.
METRIC_KEYS = [
    "IoU (soil)",
    "IoU (crop)",
    "IoU (weed)",
    "PQ (crop)",
    "PQ (weed)",
    "PQ (leaf)",
    "PQ",
    "PQ+",
]

_model_cache = {}


def _register_phenobench_metadata():
    """Register the metadata entries the model expects, without touching disk.

    The model's ``from_config`` reads ``MetadataCatalog.get(cfg.DATASETS.TRAIN[0])``.
    We only need the metadata dict to be present; we do not need the real dataset
    to be accessible, so we skip ``DatasetCatalog`` entirely.
    """
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


def get_model(name):
    """Build & load a model on first use, cache it afterwards."""
    if name in _model_cache:
        return _model_cache[name]

    info = MODELS[name]
    if not os.path.exists(info["weights"]):
        raise FileNotFoundError(
            f"Weights for model '{name}' not found at {info['weights']}."
        )

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(info["config"])
    cfg.MODEL.WEIGHTS = info["weights"]
    # Honour the CPU fallback used when running on a machine without CUDA.
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    _register_phenobench_metadata()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(info["weights"])
    _model_cache[name] = model
    return model


def get_metrics(name):
    """Return the latest evaluation metrics line written by detectron2."""
    info = MODELS[name]
    path = os.path.join(info["output_dir"], "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "IoU (crop)" in entry:
            return {
                "iteration": entry.get("iteration"),
                "values": {k: entry[k] for k in METRIC_KEYS if k in entry},
            }
    return None


def run_inference(model, image_pil):
    image = np.array(image_pil.convert("RGB"))
    H, W = image.shape[:2]
    image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32))

    with torch.no_grad():
        outputs = model([{
            "image": image_tensor,
            "image_name": "upload.png",
            "height": H,
            "width": W,
        }])

    plant_panoptic, _ = outputs[0]["plant_panoptic_seg"]
    leaf_panoptic, _ = outputs[0]["leaf_panoptic_seg"]
    plant_panoptic = plant_panoptic.cpu().numpy()
    leaf_panoptic = leaf_panoptic.cpu().numpy()

    colormap = plt.colormaps["Set1"].colors

    plant_viz = np.zeros((H, W, 3), dtype=np.uint8)
    for i, plant_id in enumerate(np.unique(plant_panoptic)):
        if plant_id != 0:
            color = tuple(int(c * 255) for c in colormap[i % 9])
            plant_viz[plant_panoptic == plant_id] = color

    leaf_viz = np.zeros((H, W, 3), dtype=np.uint8)
    for i, leaf_id in enumerate(np.unique(leaf_panoptic)):
        if leaf_id != 0:
            color = tuple(int(c * 255) for c in colormap[i % 9])
            leaf_viz[leaf_panoptic == leaf_id] = color

    overlay = (0.5 * image + 0.5 * plant_viz).astype(np.uint8)
    leaf_overlay = (0.5 * image + 0.5 * leaf_viz).astype(np.uint8)
    return image, plant_viz, leaf_viz, overlay, leaf_overlay


def to_data_uri(arr):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Server-side state so predictions accumulate across runs on the same image.
# A GET (fresh load or refresh) resets it to the initial state.
STATE = {"image": None, "image_name": None, "results": {}}


@app.route("/", methods=["GET", "POST"])
def index():
    model_choices = [(key, info["label"]) for key, info in MODELS.items()]
    error = None

    if request.method == "GET":
        # A fresh load or a page refresh returns to the initial, empty state.
        STATE["image"] = None
        STATE["image_name"] = None
        STATE["results"] = {}
    else:
        upload = request.files.get("image")
        selected = [k for k in MODELS if k in request.form.getlist("models")]

        # A newly uploaded image starts a fresh comparison; otherwise the
        # previously uploaded image is reused so the user need not re-upload.
        if upload is not None and upload.filename:
            try:
                pil = Image.open(upload.stream)
                pil.load()
                STATE["image"] = pil
                STATE["image_name"] = upload.filename
                STATE["results"] = {}
            except Exception as exc:
                error = f"Could not read the uploaded image: {exc}"

        if error is None:
            if STATE["image"] is None:
                error = "Please choose an image file before submitting."
            elif not selected:
                error = "Please select at least one model before submitting."
            else:
                # Run only the newly selected models; keep the ones already
                # computed so results accumulate on the same image.
                for key in selected:
                    if key in STATE["results"]:
                        continue
                    entry = {"key": key, "label": MODELS[key]["label"],
                             "images": None, "metrics": None, "error": None}
                    try:
                        model = get_model(key)
                        orig, plant_viz, leaf_viz, overlay, leaf_overlay = run_inference(model, STATE["image"])
                        entry["images"] = {
                            "original": to_data_uri(orig),
                            "plant": to_data_uri(plant_viz),
                            "leaf": to_data_uri(leaf_viz),
                            "overlay": to_data_uri(overlay),
                            "leaf_overlay": to_data_uri(leaf_overlay),
                        }
                        entry["metrics"] = get_metrics(key)
                    except Exception as exc:
                        entry["error"] = f"Inference failed for {entry['label']}: {exc}"
                    STATE["results"][key] = entry

    # Accumulated results and the models already run, in registry order.
    results = [STATE["results"][k] for k in MODELS if k in STATE["results"]] or None
    selected_models = [k for k in MODELS if k in STATE["results"]]

    return render_template(
        "index.html",
        model_choices=model_choices,
        selected_models=selected_models,
        results=results,
        metric_keys=METRIC_KEYS,
        error=error,
        filename=STATE["image_name"],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
