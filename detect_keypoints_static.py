from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap

# === Parameters ===
image_path = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Test_Images/towel6.png"
checkpoint_path = "model.ckpt"
network_image_size = 256
heatmap_threshold = 25
n_keypoints = 4
inward_margin = 10  # pixels to move inside from the edge

# === Load and center-crop image ===
img = Image.open(image_path).convert("RGB")
W, H = img.size
min_side = min(W, H)
left = (W - min_side) // 2
top = (H - min_side) // 2
img_cropped = img.crop((left, top, left + min_side, top + min_side))

# === Preprocessing ===
to_tensor = T.ToTensor()
resize_transform = T.Resize((network_image_size, network_image_size))
img_tensor = to_tensor(resize_transform(img_cropped)).unsqueeze(0)

# === Load model ===
model = KeypointDetector.load_from_checkpoint(
    checkpoint_path, map_location="cpu", backbone_type="Unet"
)
model.eval()

# === Inference ===
with torch.no_grad():
    heatmaps = model(img_tensor)
heatmap = heatmaps[0, 0]

# === Detect keypoints ===
keypoints = get_keypoints_from_heatmap(heatmap, heatmap_threshold, n_keypoints)
print("Detected keypoints:", keypoints)

# === Visualization ===
img_np = img_tensor[0].permute(1, 2, 0).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Draw keypoints in yellow
for kp in keypoints:
    x, y = int(kp[0]), int(kp[1])
    cv2.circle(img_bgr, (x, y), 5, (0, 255, 255), -1)

# === Compute grasping points ===
if len(keypoints) == 4:
    keypoints = np.array(keypoints)

    # Sort keypoints clockwise starting from top-left
    s = keypoints.sum(axis=1)
    diff = np.diff(keypoints, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = keypoints[np.argmin(s)]  # Top-left
    ordered[2] = keypoints[np.argmax(s)]  # Bottom-right
    ordered[1] = keypoints[np.argmin(diff)]  # Top-right
    ordered[3] = keypoints[np.argmax(diff)]  # Bottom-left

    # Compute midpoints of left and right edges
    left_edge_mid = (ordered[0] + ordered[3]) / 2
    right_edge_mid = (ordered[1] + ordered[2]) / 2

    # Vector from right to left (unit vector)
    edge_vec = left_edge_mid - right_edge_mid
    norm = np.linalg.norm(edge_vec)
    if norm > 0:
        unit_vec = edge_vec / norm
    else:
        unit_vec = np.array([0, 0])

    # Offset points slightly inward from the edge (10px)
    grasp1 = left_edge_mid - unit_vec * inward_margin
    grasp2 = right_edge_mid + unit_vec * inward_margin

    # Draw red grasping points
    for gp in [grasp1, grasp2]:
        x, y = int(gp[0]), int(gp[1])
        cv2.circle(img_bgr, (x, y), 6, (0, 0, 255), -1)

    # Print coordinates
    print(f"Grasping Point 1 (left, inward): {grasp1}")
    print(f"Grasping Point 2 (right, inward): {grasp2}")

else:
    print("Exactly 4 keypoints are required.")

# === Show image ===
cv2.imshow("Keypoints + Grasping Points", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()





