import numpy as np
import os
from phenobench import PhenoBench
from PIL import Image
from scipy.ndimage import distance_transform_edt as eucl_distance
from tqdm import tqdm

OUTPUT_DIR = '/nvmedrive/dist_maps'
DATA_DIR = '/nvmedrive/PhenoBench'

def convert_to_unsigned(matrix):
    sign_bit = np.where(matrix < 0, np.uint16(1 << 15), np.uint16(0))
    unsigned_matrix = np.abs(matrix).astype(np.uint16) + sign_bit
    return unsigned_matrix

def calculate_dist_map(posmask):
    negmask = ~posmask
    signed = eucl_distance(negmask) * negmask - (eucl_distance(posmask) - 1) * posmask
    signed = signed.astype(np.int16)
    return Image.fromarray(convert_to_unsigned(signed))

def process_image(image):
    name = image["image_name"].split('.')[0]
    pan_seg = {
        "plant": image["plant_instances"].astype(np.int32),
        "leaf":  image["leaf_instances"].astype(np.int32),
    }
    sem_seg = image["semantics"].astype(np.int32)

    for level in ["plant", "leaf"]:
        out_dir = os.path.join(OUTPUT_DIR, name, level, 'original')
        os.makedirs(out_dir, exist_ok=True)
        pan = pan_seg[level]

        # background mask (id=0)
        path = os.path.join(out_dir, '0.png')
        if not os.path.exists(path):
            calculate_dist_map(pan == 0).save(path)

        if level == "plant":
            for label in [1, 2]:
                for pid in np.unique(pan[sem_seg == label]):
                    if pid == 0:
                        continue
                    path = os.path.join(out_dir, f'{pid}.png')
                    if not os.path.exists(path):
                        calculate_dist_map(pan == pid).save(path)
        else:
            for lid in np.unique(pan):
                if lid == 0:
                    continue
                path = os.path.join(out_dir, f'{lid}.png')
                if not os.path.exists(path):
                    calculate_dist_map(pan == lid).save(path)

print("Loading dataset...")
train_data = PhenoBench(
    DATA_DIR, split='train',
    target_types=["semantics", "plant_instances", "leaf_instances"],
    make_unique_ids=False
)

print(f"Generating distance maps for {len(train_data)} images...")
for image in tqdm(train_data):
    process_image(image)

print("Done!")