#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import torch
import numpy as np

# --- FIX: MATPLOTLIB BACKEND ---
# This must go BEFORE importing pyplot.
# "Agg" is a non-interactive backend to save files without using windows (avoids Qt/xcb error).
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ====== USER CONFIG ======
BASE_DIR = "images"
WEIGHTS_PATH = "best_heatmap.pth"
IMG_SIZE = 256     # Model input size
ORIG_SIZE = 640    # Original image size (from Blender)
TOPK = 50
THRESH = 0.15
REFINE_WIN = 3
NMS_RADIUS = 50
MAX_POINTS = 4

# --- FILTER ---
FILTER_OUTLIERS = True
OUTLIER_STD_THRESHOLD = 0.95  # The lesser, the stricter
# =========================

# Import the model structure
try:
    from keypoint_heatmap_single_en import HeatmapNet
except ImportError:
    print("[ERROR] Ensure 'keypoint_heatmap_single_en.py' is in the current directory.")
    raise

def load_model(weights_path, device):
    """Loads HeatmapNet model and weights."""
    model = HeatmapNet(out_ch=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def softargmax_refine(hm, x, y, win=3):
    H, W = hm.shape
    x, y = int(x), int(y)
    x0, x1 = max(0, x - win//2), min(W, x + win//2 + 1)
    y0, y1 = max(0, y - win//2), min(H, y + win//2 + 1)
    patch = hm[y0:y1, x0:x1]
    if patch.size == 0 or patch.sum() <= 1e-8:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    s = patch.sum()
    return (patch * xx).sum() / s, (patch * yy).sum() / s

def topk_peaks(hm, k=50, thresh=0.35, refine_win=3):
    flat = hm.reshape(-1)
    idxs = np.argsort(-flat)
    peaks = []
    H, W = hm.shape
    for idx in idxs:
        score = float(flat[idx])
        if score < thresh: break
        y, x = divmod(idx, W)
        rx, ry = softargmax_refine(hm, x, y, win=refine_win)
        peaks.append((rx, ry, score))
        if len(peaks) == k: break
    return peaks

def image_space_nms(peaks_img, radius_px=20, max_points=4):
    if not peaks_img: return []
    peaks = sorted(peaks_img, key=lambda p: p[2], reverse=True)
    kept = []
    for x, y, sc in peaks:
        suppress = False
        for kx, ky, _ in kept:
            if (x - kx)**2 + (y - ky)**2 <= radius_px**2:
                suppress = True
                break
        if not suppress:
            kept.append((x, y, sc))
        if len(kept) >= max_points: break
    return kept

# --- NEW FILTERING FUNCTION ---
def filter_statistical_outliers(points, std_thresh=1.5):
    """
    Removes points that are statistically far from the group centroid.
    """
    # We need at least 3 points to determine a meaningful outlier
    if len(points) < 3:
        return points

    # Extract X, Y coordinates
    coords = np.array([[p[0], p[1]] for p in points])
    
    # 1. Calculate centroid (average position)
    centroid = np.mean(coords, axis=0)
    
    # 2. Calculate distances from each point to the centroid
    distances = np.linalg.norm(coords - centroid, axis=1)
    
    # 3. Calculate mean and standard deviation of distances
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # If deviation is very small, points are very close (clustered), do not remove anything
    if std_dist < 10.0:
        return points

    valid_points = []
    for i, p in enumerate(points):
        # Condition: Distance must be less than Mean + (Threshold * StdDev)
        # If a point is very far, its distance will be much larger than the average.
        limit = mean_dist + (std_thresh * std_dist)
        
        if distances[i] <= limit:
            valid_points.append(p)
        # else: print(f"Outlier removed: {p}") # Debug

    return valid_points

def main():
    device = torch.device("cpu")
    print(f"Loading model from {WEIGHTS_PATH} to {device}...")
    model = load_model(WEIGHTS_PATH, device)

    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] Directory '{BASE_DIR}' not found. Run scene-generator.py first.")
        return

    scene_folders = sorted([f for f in os.listdir(BASE_DIR) if f.startswith("scene")])
    if not scene_folders:
        print(f"[WARNING] No scene folders found in '{BASE_DIR}'.")

    for scene_name in scene_folders:
        scene_path = os.path.join(BASE_DIR, scene_name)
        raw_images_path = os.path.join(scene_path, "raw_images")
        gt_json_path = os.path.join(scene_path, "keypoints_gt.json")
        out_kp_dir = os.path.join(scene_path, "keypoints")
        os.makedirs(out_kp_dir, exist_ok=True)
        
        gt_data = {}
        if os.path.exists(gt_json_path):
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)
        
        results_pred = {}
        print(f"Processing {scene_name}...")
        
        if not os.path.exists(raw_images_path):
            continue

        for fname in sorted(os.listdir(raw_images_path)):
            if not fname.endswith(("_raw.png", ".png", ".jpg")):
                continue
            
            img_path = os.path.join(raw_images_path, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            # --- Inference ---
            resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                logits = model(t)
                hm = torch.sigmoid(logits)[0, 0].cpu().numpy()
            
            peaks = topk_peaks(hm, k=TOPK, thresh=THRESH, refine_win=REFINE_WIN)
            
            h_hm, w_hm = hm.shape
            Sx, Sy = ORIG_SIZE / w_hm, ORIG_SIZE / h_hm
            peaks_img = [(p[0]*Sx, p[1]*Sy, p[2]) for p in peaks]
            
            # NMS
            final_peaks = image_space_nms(peaks_img, radius_px=NMS_RADIUS, max_points=MAX_POINTS)
            
            # --- NEW: STATISTICAL FILTER ---
            if FILTER_OUTLIERS:
                final_peaks = filter_statistical_outliers(final_peaks, std_thresh=OUTLIER_STD_THRESHOLD)

            # Store predictions
            pred_coords = [[float(p[0]), float(p[1])] for p in final_peaks]
            results_pred[fname] = pred_coords
            
            # --- Visualization ---
            plt.figure(figsize=(8, 8))
            img_rgb_plt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb_plt)
            
            if fname in gt_data:
                gt_pts = gt_data[fname]
                gt_x = [p[0] for p in gt_pts]
                gt_y = [p[1] for p in gt_pts]
                plt.scatter(gt_x, gt_y, c='lime', s=80, marker='o', label='Ground truths', edgecolors='black')
                
            if pred_coords:
                pred_x = [p[0] for p in pred_coords]
                pred_y = [p[1] for p in pred_coords]
                plt.scatter(pred_x, pred_y, c='red', s=80, marker='x', linewidths=3, label='Predicted points')
            
            plt.legend(loc='upper right')
            plt.axis('off')
            
            viz_name = fname.replace("_raw.png", "_viz.png")
            plt.savefig(os.path.join(out_kp_dir, viz_name), bbox_inches='tight', pad_inches=0)
            plt.close()
            
        with open(os.path.join(out_kp_dir, "keypoints_pred.json"), "w") as f:
            json.dump(results_pred, f, indent=4)
            
    print("Done processing keypoints.")

if __name__ == "__main__":
    main()