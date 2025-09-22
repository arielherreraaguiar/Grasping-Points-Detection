import os
import json
import random
import shutil
import zipfile
from glob import glob
from PIL import Image, ImageEnhance
import numpy as np

# Paths
base_training = "wrinkle-training"
base_final = "wrinkle-dataset-final"
output_zip = "wrinkle-data-final.zip"

# Create output dirs if not exist
for split in ["train", "test", "valid"]:
    os.makedirs(os.path.join(base_final, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_final, split, "labels"), exist_ok=True)

# --- Augmentation functions ---
def augment_image(img):
    """Apply random augmentation to the image"""
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)

    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img

# --- Convert JSON to YOLO txt ---
def json_to_yolo(json_path, img_w, img_h):
    with open(json_path, "r") as f:
        data = json.load(f)

    yolo_lines = []
    for shape in data["shapes"]:
        if shape["label"] != "wrinkle":
            continue
        points = np.array(shape["points"])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Convert to YOLO normalized format
        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        yolo_lines.append(f"0 {x_center} {y_center} {w} {h}")

    return "\n".join(yolo_lines)

# --- Collect training images ---
image_paths = sorted(glob(os.path.join(base_training, "*.png")))
json_paths = sorted(glob(os.path.join(base_training, "*.json")))

dataset = []

# Augmentation loop
target_size = 500
while len(dataset) < target_size:
    for img_path, js_path in zip(image_paths, json_paths):
        img = Image.open(img_path).convert("RGB")
        aug_img = augment_image(img)

        # Save augmented image temporarily
        new_name = f"{len(dataset):06d}.jpg"
        img_w, img_h = aug_img.size

        # Convert JSON to YOLO
        yolo_txt = json_to_yolo(js_path, img_w, img_h)

        dataset.append((aug_img, yolo_txt, new_name))
        if len(dataset) >= target_size:
            break

# --- Split dataset ---
random.shuffle(dataset)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_test = int(0.1 * n_total)

splits = {
    "train": dataset[:n_train],
    "test": dataset[n_train:n_train+n_test],
    "valid": dataset[n_train+n_test:]
}

# Save images and labels
for split, items in splits.items():
    for img, label_txt, fname in items:
        img.save(os.path.join(base_final, split, "images", fname))
        with open(os.path.join(base_final, split, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.write(label_txt)

# --- Create final zip ---
shutil.make_archive("wrinkle-data-final", 'zip', base_final)
print(f"✅ Dataset ready and zipped as {output_zip}")
