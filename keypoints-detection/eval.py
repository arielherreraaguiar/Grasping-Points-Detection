#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict keypoints on a single image with HeatmapNet and display them.
- Loads one image given by image_path
- Runs inference on CPU
- Detects peaks from the heatmap
- Draws predicted points (red) and prints their coordinates
"""

import os
import sys
import cv2
import json
import torch
import numpy as np

# ====== USER CONFIG ======
# Put your absolute image path here (or pass as CLI arg)
image_path = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Towel-scenes-custom/final_dataset/test/000468_rgb.png"

# Model weights and input size
WEIGHTS    = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/best_heatmap.pth"
IMG_SIZE   = 256

# Peak detection params
TOPK       = 4       # max number of points to return
THRESH     = 0.35    # confidence threshold in [0,1]
REFINE_WIN = 3       # softargmax window size (odd number)
# =========================

try:
    # Make sure this import path matches your project structure
    from keypoint_heatmap_single_en import HeatmapNet
except Exception as e:
    print("[ERROR] Import HeatmapNet failed:", e)
    sys.exit(1)

# ---------- Model loader ----------
def load_model(weights_path, device):
    """Load HeatmapNet and weights."""
    model = HeatmapNet(out_ch=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

# ---------- Softargmax refine ----------
def softargmax_refine(hm, x, y, win=3):
    """Refine integer peak (x,y) using a local soft-argmax on heatmap hm."""
    H, W = hm.shape
    x, y = int(x), int(y)
    x0, x1 = max(0, x - win//2), min(W, x + win//2 + 1)
    y0, y1 = max(0, y - win//2), min(H, y + win//2 + 1)
    patch = hm[y0:y1, x0:x1]
    if patch.size == 0:
        return float(x), float(y)
    s = patch.sum()
    if s <= 1e-8:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rx = (patch * xx).sum() / s
    ry = (patch * yy).sum() / s
    return float(rx), float(ry)

# ---------- Peak detector ----------
def topk_peaks(hm, k=4, thresh=0.35, refine_win=3):
    """Return up to k peaks as (x, y, score) in heatmap coordinates."""
    H, W = hm.shape
    flat = hm.reshape(-1)
    idxs = np.argsort(-flat)  # descending
    peaks = []
    for idx in idxs:
        score = float(flat[idx])
        if score < thresh:
            break
        y, x = divmod(idx, W)
        rx, ry = softargmax_refine(hm, x, y, win=refine_win)
        peaks.append((rx, ry, score))
        if len(peaks) == k:
            break
    return peaks

# ---------- Main single-image prediction ----------
def main():
    # Allow path by CLI: python predict_points_single.py /abs/path/to/img.png
    img_path = sys.argv[1] if len(sys.argv) > 1 else image_path
    if not os.path.isfile(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        sys.exit(1)

    # Keep original size for overlay scaling back if needed
    H0, W0 = img_bgr.shape[:2]

    # Resize to model input
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Preprocess to tensor
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float()

    device = torch.device("cpu")
    model = load_model(WEIGHTS, device)

    # Inference
    with torch.no_grad():
        logits = model(t.to(device))
        hm = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H,W) heatmap in [0,1]

    # Peaks in heatmap coords -> scale to IMG_SIZE coords
    peaks = topk_peaks(hm, k=TOPK, thresh=THRESH, refine_win=REFINE_WIN)
    h, w = hm.shape
    sx, sy = IMG_SIZE / float(w), IMG_SIZE / float(h)
    pred_px_imgsize = np.array([[x * sx, y * sy] for (x, y, _) in peaks], dtype=np.float32)

    # If you want to map back to the original image size:
    Sx = W0 / float(IMG_SIZE)
    Sy = H0 / float(IMG_SIZE)
    pred_px_orig = np.array([[p[0] * Sx, p[1] * Sy] for p in pred_px_imgsize], dtype=np.float32)

    # Print coordinates
    print("Predicted points (IMG_SIZE space, origin top-left):")
    for i, (x, y) in enumerate(pred_px_imgsize):
        print(f"  #{i+1}: x={x:.1f}, y={y:.1f}")

    print("\nPredicted points (ORIGINAL image space):")
    for i, (x, y) in enumerate(pred_px_orig):
        print(f"  #{i+1}: x={x:.1f}, y={y:.1f}")

    # Visualize on original-size image
    vis = img_bgr.copy()
    for (x, y) in pred_px_orig:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)  # red predicted

    win_name = "Predicted Points (red)"
    cv2.imshow(win_name, vis)
    print("\nPress any key on the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
