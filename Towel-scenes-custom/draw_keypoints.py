import os
import cv2
import json

# Paths
image_dir = "images"
json_path = os.path.join(image_dir, "dataset.json")
output_dir = "corners"
os.makedirs(output_dir, exist_ok=True)

# Load annotations
with open(json_path, "r") as f:
    data = json.load(f)

# Draw red dots on each image based on keypoints
for item in data["images"]:
    img_path = os.path.join(image_dir, item["file_name"])
    output_path = os.path.join(output_dir, item["file_name"])

    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    keypoints = item.get("keypoints", [])
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(output_path, img)

print("All keypoints drawn successfully and saved in 'corners/' folder.")
