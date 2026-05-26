"""
Compute repeat factor for each training image based on rare class presence.
Images containing weeds (rare class) get a higher repeat factor.
Used by RepeatFactorTrainingSampler to balance the training distribution.

Output: repeat_factors.json (image_name -> float)

Usage:
    python compute_repeat_factors.py
"""
import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET_ROOT = "/nvmedrive/PhenoBenchExtra"
SPLIT = "train"
OUTPUT_FILE = "repeat_factors.json"
# THRESHOLD controls how aggressive the oversampling is.
# Set this CLOSE to the actual rare class frequency to avoid over-oversampling.
# Examples (weed freq ~= 0.005 in PhenoBench):
#   - 0.005 -> rf = 1.0  (no oversampling)
#   - 0.010 -> rf ~= 1.4 (mild, 1.4x training time)
#   - 0.020 -> rf ~= 2.0 (moderate, 2x training time)
#   - 0.100 -> rf ~= 4.5 (aggressive, 4.5x training time)
THRESHOLD = 0.010  # mild oversampling: ~1.4x training time

semantics_dir = os.path.join(DATASET_ROOT, SPLIT, "semantics")
files = sorted(os.listdir(semantics_dir))

# Step 1: Compute global pixel frequency per class across the dataset
print(f"Found {len(files)} training images.\n")
print("Step 1: Computing global class frequencies...")
class_pixels = {0: 0, 1: 0, 2: 0}  # soil, crop, weed
total_pixels = 0

for f in tqdm(files):
    sem = np.array(Image.open(os.path.join(semantics_dir, f)))
    # Remap partial annotations to main classes (matches register_phenobench.py logic)
    sem[sem == 3] = 1  # partial_crop -> crop
    sem[sem == 4] = 2  # partial_weed -> weed

    for cls in [0, 1, 2]:
        class_pixels[cls] += int(np.sum(sem == cls))
    total_pixels += sem.size

class_freq = {c: class_pixels[c] / total_pixels for c in [0, 1, 2]}
print(f"\nClass pixel frequencies:")
print(f"  soil: {class_freq[0]:.4f} ({class_freq[0]*100:.2f}%)")
print(f"  crop: {class_freq[1]:.4f} ({class_freq[1]*100:.2f}%)")
print(f"  weed: {class_freq[2]:.4f} ({class_freq[2]*100:.2f}%)")

# Step 2: Compute per-image repeat factor based on rare class presence
print("\nStep 2: Computing repeat factors per image...")
repeat_factors = {}
images_with_weed = 0
images_with_crop = 0

for f in tqdm(files):
    sem = np.array(Image.open(os.path.join(semantics_dir, f)))
    sem[sem == 3] = 1
    sem[sem == 4] = 2

    # For each "thing" class (crop, weed) present in the image,
    # compute repeat factor = max(1.0, sqrt(threshold / class_freq))
    rf = 1.0
    has_weed = bool(np.any(sem == 2))
    has_crop = bool(np.any(sem == 1))

    if has_weed:
        images_with_weed += 1
    if has_crop:
        images_with_crop += 1

    # Per-image significance threshold: ignore tiny presence of a class.
    # Class is "significant" only if it covers >= 0.5% of the image pixels.
    MIN_PIXEL_FRACTION = 0.005
    img_pixels = sem.size

    for cls in [1, 2]:  # Skip soil (always present, never rare)
        cls_pixels = int(np.sum(sem == cls))
        if cls_pixels / img_pixels >= MIN_PIXEL_FRACTION:
            class_rf = np.sqrt(THRESHOLD / max(class_freq[cls], 1e-6))
            rf = max(rf, class_rf)

    repeat_factors[f] = float(rf)

# Save to JSON
with open(OUTPUT_FILE, 'w') as fp:
    json.dump(repeat_factors, fp, indent=2)

# Print statistics
rf_values = list(repeat_factors.values())
print(f"\nSummary:")
print(f"  Total training images: {len(files)}")
print(f"  Images with crop:      {images_with_crop} ({images_with_crop/len(files)*100:.1f}%)")
print(f"  Images with weed:      {images_with_weed} ({images_with_weed/len(files)*100:.1f}%)")
print(f"\nRepeat factor distribution:")
print(f"  Min:  {min(rf_values):.3f}")
print(f"  Max:  {max(rf_values):.3f}")
print(f"  Mean: {np.mean(rf_values):.3f}")
print(f"  Images with rf > 1.5: {sum(1 for v in rf_values if v > 1.5)}")
print(f"  Images with rf > 2.0: {sum(1 for v in rf_values if v > 2.0)}")
print(f"\nSaved to: {OUTPUT_FILE}")
