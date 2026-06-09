"""
Per-image plant count evaluation: how many crop and weed instances per image
in GT vs predictions.

Usage:
    python count_plants.py --phenobench_dir /nvmedrive/PhenoBench \
                           --prediction_dir /tmp/out2 \
                           --split val

Outputs (per class — crop and weed, separately):
  - total counts (GT and predicted) across all images
  - average / median / min / max plants per image
  - per-image MAE and RMSE of the count
  - exact-match rate (fraction of images with predicted == GT)
  - distribution by bucket (0 / 1-2 / 3-5 / 6-10 / 11-20 / 21+)
"""
import argparse
import math
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from phenobench.evaluation.auxiliary.common import (
    get_png_files_in_dir,
    load_file_as_tensor,
    load_file_as_int_tensor,
)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phenobench_dir', required=True, type=Path,
                        help='Path to ground truth directory.')
    parser.add_argument('--prediction_dir', required=True, type=Path,
                        help='Path to prediction directory.')
    parser.add_argument('--split', default='val', type=str,
                        help='Specify which split to use for evaluation.')
    return vars(parser.parse_args())


def count_instances(plant_map, semantics, class_label):
    """Count distinct plant instance IDs whose pixels belong to class_label."""
    if hasattr(plant_map, 'numpy'):
        plant_map = plant_map.numpy()
    if hasattr(semantics, 'numpy'):
        semantics = semantics.numpy()
    ids = np.unique(plant_map[semantics == class_label])
    ids = ids[ids != 0]   # background id 0 is not an instance
    return int(len(ids))


def _summary(label, counts):
    if not counts:
        print(f'{label}: no data')
        return
    arr = np.array(counts)
    print(f'{label}: total={int(arr.sum())}  imgs={len(arr)}  '
          f'avg={arr.mean():.2f}  median={float(np.median(arr)):.1f}  '
          f'min={int(arr.min())}  max={int(arr.max())}')


def _hist(counts):
    if not counts:
        return None
    buckets = {'0': 0, '1-2': 0, '3-5': 0, '6-10': 0, '11-20': 0, '21+': 0}
    for c in counts:
        if c == 0:
            buckets['0'] += 1
        elif c <= 2:
            buckets['1-2'] += 1
        elif c <= 5:
            buckets['3-5'] += 1
        elif c <= 10:
            buckets['6-10'] += 1
        elif c <= 20:
            buckets['11-20'] += 1
        else:
            buckets['21+'] += 1
    return buckets


def _print_hist(label, gt_counts, pred_counts):
    gt_hist = _hist(gt_counts)
    pred_hist = _hist(pred_counts)
    print(f'--- Per-image plant-count distribution: {label} (#images per bucket) ---')
    if not gt_hist or not pred_hist:
        print('  (no data)')
        return
    print(f'{"bucket":>8}  {"GT":>8}  {"Pred":>8}')
    for key in gt_hist:
        print(f'{key:>8}  {gt_hist[key]:>8}  {pred_hist[key]:>8}')


def main():
    args = parse_args()

    gt_plant_fnames = get_png_files_in_dir(
        args['phenobench_dir'] / args['split'] / 'plant_instances')
    gt_sem_fnames = get_png_files_in_dir(
        args['phenobench_dir'] / args['split'] / 'semantics')
    pred_plant_fnames = get_png_files_in_dir(
        args['prediction_dir'] / 'plant_instances')
    pred_sem_fnames = get_png_files_in_dir(
        args['prediction_dir'] / 'semantics')

    n_total = len(gt_plant_fnames)

    # Per-image counts
    gt_crop_counts = []
    pred_crop_counts = []
    gt_weed_counts = []
    pred_weed_counts = []

    for gt_plant_f, gt_sem_f, pred_plant_f, pred_sem_f in tqdm(zip(
            gt_plant_fnames, gt_sem_fnames, pred_plant_fnames, pred_sem_fnames),
            total=n_total):

        assert gt_plant_f == gt_sem_f

        gt_plant = load_file_as_tensor(
            args['phenobench_dir'] / args['split'] / 'plant_instances' / gt_plant_f).squeeze()
        gt_sem = load_file_as_tensor(
            args['phenobench_dir'] / args['split'] / 'semantics' / gt_sem_f).squeeze()
        pred_plant = load_file_as_int_tensor(
            args['prediction_dir'] / 'plant_instances' / pred_plant_f).squeeze()
        pred_sem = load_file_as_int_tensor(
            args['prediction_dir'] / 'semantics' / pred_sem_f).squeeze()

        gt_crop_counts.append(count_instances(gt_plant, gt_sem, class_label=1))
        pred_crop_counts.append(count_instances(pred_plant, pred_sem, class_label=1))
        gt_weed_counts.append(count_instances(gt_plant, gt_sem, class_label=2))
        pred_weed_counts.append(count_instances(pred_plant, pred_sem, class_label=2))

    # ---- Aggregate metrics per class ----
    def report(label, gt_counts, pred_counts):
        gt = np.array(gt_counts, dtype=np.float64)
        pred = np.array(pred_counts, dtype=np.float64)
        diff = pred - gt
        mae = float(np.mean(np.abs(diff)))
        rmse = float(math.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))                       # >0 if model over-counts
        exact = int(np.sum(diff == 0))
        exact_rate = exact / len(gt) if len(gt) > 0 else 0.0

        print()
        print(f'--- {label} count statistics ---')
        _summary('GT       ', gt.astype(int).tolist())
        _summary('Predicted', pred.astype(int).tolist())
        print(f'MAE  per image: {mae:.3f} ({label.lower()}s)')
        print(f'RMSE per image: {rmse:.3f} ({label.lower()}s)')
        print(f'Bias (Pred - GT) per image: {bias:+.3f} ({label.lower()}s)')
        print(f'Exact-count images: {exact}/{len(gt)} ({exact_rate*100:.2f}%)')

    report('CROP', gt_crop_counts, pred_crop_counts)
    report('WEED', gt_weed_counts, pred_weed_counts)

    print()
    _print_hist('CROP', gt_crop_counts, pred_crop_counts)
    print()
    _print_hist('WEED', gt_weed_counts, pred_weed_counts)


if __name__ == '__main__':
    main()
