import os
import cv2
import json

# Paths
image_dir = "images"
json_path = os.path.join(image_dir, "dataset_coco.json")
output_dir = "corners"
os.makedirs(output_dir, exist_ok=True)

# Load COCO-style annotations
with open(json_path, "r") as f:
    data = json.load(f)

# Create image_id to filename mapping
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Loop through each annotation
for ann in data["annotations"]:
    image_id = ann["image_id"]
    filename = image_id_to_filename.get(image_id)
    if not filename:
        continue

    img_path = os.path.join(image_dir, filename)
    output_path = os.path.join(output_dir, filename)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    keypoints = ann.get("keypoints", [])
    # COCO format: [x1, y1, v1, x2, y2, v2, ..., xk, yk, vk]
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]
        if v > 0:  # 1 = labeled but not visible, 2 = visible
            cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(output_path, img)

print("All visible keypoints drawn and saved in 'corners/' folder.")
