# Hierarchical Mask2Former — Panoptic Segmentation of Crops, Weeds and Leaves

This repository extends the [Hierarchical Mask2Former](https://github.com/facebookresearch/Mask2Former) method for the joint plant-level and leaf-level panoptic segmentation of the [PhenoBench](https://www.phenobench.org) dataset. On top of the original method it adds a Tversky loss, a stronger data-augmentation and class-rebalancing pipeline, and a self-supervised [Hiera](https://github.com/facebookresearch/hiera) backbone, together with evaluation and instance-counting scripts and an interactive web application for inference.

The work was carried out as part of a master's dissertation by **Melinda Henrietta Incze**.

## Features

- Hierarchical panoptic segmentation at two levels: plants (crop / weed) and leaves.
- Three interchangeable backbones: ResNet-50, Swin-L (ImageNet-21k) and Hiera-L (self-supervised, MAE).
- Tversky loss, boundary loss, class-balanced sampling and full augmentation.
- Evaluation of semantic IoU, Panoptic Quality (PQ, SQ, RQ) and instance counting.
- A Flask web application for uploading an image and inspecting the predictions of each model.

## Installation

The code was developed and tested on Linux / WSL2 with Python 3.8 and a CUDA-capable GPU.

1. Create the environment (conda is recommended):

   ```bash
   conda env create -f environment.yml
   conda activate mask2former
   # or, with pip:
   pip install -r requirements.txt
   ```

2. Install [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html) matching your PyTorch and CUDA versions.

3. Install `timm` (required by the Hiera backbone):

   ```bash
   pip install -U timm
   ```

4. Compile the MSDeformAttn CUDA kernel used by the pixel decoder:

   ```bash
   cd mask2former/modeling/pixel_decoder/ops
   sh make.sh
   ```

See [INSTALL.md](INSTALL.md) for the original upstream installation notes.

## Dataset

Download the PhenoBench dataset from [phenobench.org](https://www.phenobench.org). After extraction, each split contains the following folders:

```
PhenoBench/
├── train/  ├── images/  ├── semantics/  ├── plant_instances/  └── leaf_instances/
├── val/    └── ...
└── test/   └── images/
```

The dataset location is configured through the `register_phenobench(...)` calls in [train_net.py](train_net.py); set their `root` argument to your PhenoBench directory.

Semantic labels follow the PhenoBench convention (`0` soil, `1` crop, `2` weed, `3`/`4` partial crop/weed); the partial classes are remapped to their full classes during loading.

If the boundary loss is used, the signed distance maps must be generated once beforehand:

```bash
python generate_dist_maps.py
```

## Training

Train a model by selecting one of the configuration files:

```bash
python train_net.py --num-gpus 1 --config-file configs/phenobench/exp_swinL_tversky.yaml
```

The main configurations are:

| Config | Backbone | Description |
| --- | --- | --- |
| `configs/phenobench/exp_swinL_baseline.yaml` | Swin-L | original method, without this work's additions |
| `configs/phenobench/exp_swinL_tversky.yaml`  | Swin-L | proposed additions (Tversky, augmentation, sampler) |
| `configs/phenobench/exp_hiera_tversky.yaml`  | Hiera-L | self-supervised backbone with the proposed additions |

The pretrained backbone weights are loaded through `MODEL.WEIGHTS`; the Hiera backbone loads its self-supervised weights internally through `timm`.

## Evaluation

Evaluate a trained checkpoint:

```bash
python train_net.py --num-gpus 1 --config-file configs/phenobench/exp_swinL_tversky.yaml \
  --eval-only MODEL.WEIGHTS output/Exp_SwinL_Tversky/model_final.pth
```

Additional analysis scripts:

```bash
# plant and leaf instance counting (MAE, RMSE, precision / recall / F1)
python count_plants.py --phenobench_dir /path/to/PhenoBench --prediction_dir /path/to/predictions --split val
python count_leaves.py --phenobench_dir /path/to/PhenoBench --prediction_dir /path/to/predictions --split val

# dataset class distribution (class imbalance)
python plot_class_distribution.py --phenobench_dir /path/to/PhenoBench --split all
```

## Web application

An interactive Flask application allows uploading an image and comparing the predictions of the trained models. Start it from the project root:

```bash
python -m webapp.app
```

Then open [http://localhost:5000](http://localhost:5000). For each selected model the application shows the input image, the plant- and leaf-level panoptic predictions and their overlays, together with the latest evaluation metrics. The application expects the trained checkpoints under the `output/` directory (for example `output/Exp_SwinL_Tversky/model_final.pth`).

## Licenses

The code of this project is released under the MIT License.

It builds on several third-party components, each kept under its own license:

| Component | License |
| --- | --- |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | MIT |
| [Detectron2](https://github.com/facebookresearch/detectron2) | Apache-2.0 |
| [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) | MIT |
| [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | Apache-2.0 |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| [Hiera](https://github.com/facebookresearch/hiera) | code: Apache-2.0 — **pretrained weights: CC BY-NC 4.0 (non-commercial)** |

**Dataset.** The PhenoBench dataset is distributed under the **CC BY-SA 4.0** license, which requires attribution (a citation of the corresponding paper) and share-alike redistribution of any modified version.

> Note: because the Hiera pretrained weights are released under CC BY-NC 4.0, models based on the Hiera backbone are intended for non-commercial, research use only.

## Citation

If this work is useful, please cite the original Hierarchical Mask2Former method and the PhenoBench dataset:

```bibtex
@article{hierarchical2023darbyshire,
  author = {Darbyshire, Madeleine and Sklar, Elizabeth and Parsons, Simon},
  year   = {2023},
  title  = {Hierarchical Mask2Former: Panoptic Segmentation of Crops, Weeds and Leaves},
  doi    = {10.13140/RG.2.2.33051.23847}
}

@article{weyler2023phenobench,
  author  = {Weyler, Jan and Magistri, Federico and Marks, Elias and Chong, Yue Linn and Sodano, Matteo and Roggiolani, Gianmarco and Chebrolu, Nived and Stachniss, Cyrill and Behley, Jens},
  title   = {PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2024}
}
```

## Acknowledgements

This project is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Meta AI Research and uses the [PhenoBench](https://www.phenobench.org) dataset from the University of Bonn. The Hiera backbone is provided through the [timm](https://github.com/huggingface/pytorch-image-models) library.
