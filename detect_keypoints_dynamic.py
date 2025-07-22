import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap

# === Parameters ===
checkpoint_path = "model.ckpt"
network_image_size = 256
heatmap_threshold = 25
n_keypoints = 4
inward_margin = 10  # pixels inward from border

# === Load model ===
model = KeypointDetector.load_from_checkpoint(
    checkpoint_path, map_location="cpu", backbone_type="Unet"
)
model.eval()

# === Preprocessing transforms ===
resize_transform = T.Resize((network_image_size, network_image_size))
to_tensor = T.ToTensor()

# === RealSense setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Press ESC to quit.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Center crop (square)
        H, W, _ = img_rgb.shape
        min_side = min(H, W)
        top = (H - min_side) // 2
        left = (W - min_side) // 2
        cropped = img_rgb[top:top + min_side, left:left + min_side]

        # Preprocess
        img_pil = T.ToPILImage()(cropped)
        img_tensor = to_tensor(resize_transform(img_pil)).unsqueeze(0)

        # Inference
        with torch.no_grad():
            heatmaps = model(img_tensor)
        heatmap = heatmaps[0, 0]
        keypoints = get_keypoints_from_heatmap(heatmap, heatmap_threshold, n_keypoints)

        # Prepare image for drawing
        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw keypoints
        if len(keypoints) == 4:
            keypoints = np.array(keypoints)

            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(img_bgr, (x, y), 5, (0, 255, 255), -1)

            # Order keypoints clockwise
            s = keypoints.sum(axis=1)
            diff = np.diff(keypoints, axis=1).flatten()
            ordered = np.zeros((4, 2), dtype=np.float32)
            ordered[0] = keypoints[np.argmin(s)]  # Top-left
            ordered[2] = keypoints[np.argmax(s)]  # Bottom-right
            ordered[1] = keypoints[np.argmin(diff)]  # Top-right
            ordered[3] = keypoints[np.argmax(diff)]  # Bottom-left

            # Check all edge pairs and find the longest one
            edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            max_dist = 0
            longest_edge = (0, 1)
            for i, j in edge_pairs:
                dist = np.linalg.norm(ordered[i] - ordered[j])
                if dist > max_dist:
                    max_dist = dist
                    longest_edge = (i, j)

            # Get the opposite edge
            opposite_edge = ((longest_edge[0] + 2) % 4, (longest_edge[1] + 2) % 4)

            # Get midpoints
            p1, p2 = ordered[longest_edge[0]], ordered[longest_edge[1]]
            p3, p4 = ordered[opposite_edge[0]], ordered[opposite_edge[1]]

            mid1 = (p1 + p2) / 2
            mid2 = (p3 + p4) / 2

            # Vector from midpoint2 to midpoint1 (directional)
            dir_vec = mid1 - mid2
            unit_dir = dir_vec / np.linalg.norm(dir_vec) if np.linalg.norm(dir_vec) != 0 else np.array([0, 0])

            grasp1 = mid1 - unit_dir * inward_margin
            grasp2 = mid2 + unit_dir * inward_margin

            # Draw red grasping points
            for gp in [grasp1, grasp2]:
                x, y = int(gp[0]), int(gp[1])
                cv2.circle(img_bgr, (x, y), 6, (0, 0, 255), -1)

            # Print coordinates
            print(f"Grasping Point 1: {grasp1}")
            print(f"Grasping Point 2: {grasp2}")
        else:
            print("Exactly 4 keypoints are required.")

        # Show image
        cv2.imshow("RealSense Keypoints + Grasping Points", img_bgr)
        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
