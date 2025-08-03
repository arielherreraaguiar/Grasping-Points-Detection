import os
import json
import random
import shutil

# Paths
input_json = "images/dataset_coco.json"
input_img_dir = "images"
output_dir = "final_dataset"
os.makedirs(output_dir, exist_ok=True)

# Output folders
splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]

for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Load COCO-style JSON
with open(input_json, 'r') as f:
    data = json.load(f)

# All image entries
all_images = data['images']
annotations = data['annotations']
categories = data['categories']

# Shuffle and split image entries
random.seed(42)
random.shuffle(all_images)

n = len(all_images)
n_train = int(split_ratio[0] * n)
n_val = int(split_ratio[1] * n)

split_map = {
    'train': all_images[:n_train],
    'val': all_images[n_train:n_train + n_val],
    'test': all_images[n_train + n_val:]
}

# Create and save JSONs per split
for split, img_entries in split_map.items():
    # Collect image IDs in current split
    img_ids = set(img['id'] for img in img_entries)

    # Filter annotations for current split
    anns = [ann for ann in annotations if ann['image_id'] in img_ids]

    # Create split dict
    split_json = {
        'images': img_entries,
        'annotations': anns,
        'categories': categories
    }

    # Save split JSON file
    output_json_path = os.path.join(output_dir, split, 'dataset_coco.json')
    with open(output_json_path, 'w') as f:
        json.dump(split_json, f, indent=4)

    # Copy corresponding images
    for img in img_entries:
        src = os.path.join(input_img_dir, img['file_name'])
        dst = os.path.join(output_dir, split, img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Image not found: {src}")

print("Dataset split completed and saved to final_dataset/")
