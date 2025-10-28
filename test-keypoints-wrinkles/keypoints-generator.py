#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch prediction of keypoints on all images in a folder using HeatmapNet + Image-space NMS.
- Loads each image from `images/`
- Runs inference on CPU
- Detects peaks from the heatmap
- Applies Non-Maximum Suppression (NMS) in image space (avoid duplicate nearby points)
- Draws predicted points (red) and saves them in `corners/`
- Exports a JSON file with coordinates of detected keypoints
"""

import os
import cv2
import json
import torch
import numpy as np

# ====== USER CONFIG ======
IMAGES_DIR = "images"
OUTPUT_DIR = "corners"
WEIGHTS    = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/best_heatmap.pth"
IMG_SIZE   = 256
TOPK       = 50      # more candidates (before NMS)
THRESH     = 0.15
REFINE_WIN = 3
NMS_RADIUS = 50     # radius in pixels for suppression
MAX_POINTS = 4       # keep up to 4 final points
# =========================

# Import model
try:
    from keypoint_heatmap_single_en import HeatmapNet
except Exception as e:
    print("[ERROR] Import HeatmapNet failed:", e)
    raise

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
def topk_peaks(hm, k=50, thresh=0.35, refine_win=3):
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

# ---------- Image-space NMS ----------
def image_space_nms(peaks_img, radius_px=20, max_points=4):
    """
    Greedy NMS in IMAGE space.
    peaks_img: list of (x, y, score) in *original image* pixels.
    Keeps the highest-score peak, suppresses other peaks within 'radius_px'.
    """
    if len(peaks_img) == 0:
        return []
    peaks = sorted(peaks_img, key=lambda p: p[2], reverse=True)
    kept = []
    for x, y, sc in peaks:
        suppress = False
        for kx, ky, _ in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 <= radius_px ** 2:
                suppress = True
                break
        if not suppress:
            kept.append((x, y, sc))
        if len(kept) >= max_points:
            break
    return kept

# ---------- Batch prediction ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    model = load_model(WEIGHTS, device)

    results_dict = {}

    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(IMAGES_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARNING] Failed to read {fname}, skipping.")
            continue

        H0, W0 = img_bgr.shape[:2]

        # Resize to model input
        resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

        # Preprocess to tensor
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float()

        # Inference
        with torch.no_grad():
            logits = model(t.to(device))
            hm = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Detect peaks in heatmap coords
        peaks = topk_peaks(hm, k=TOPK, thresh=THRESH, refine_win=REFINE_WIN)

        # Map to original resolution
        h, w = hm.shape
        sx, sy = IMG_SIZE / float(w), IMG_SIZE / float(h)
        pred_px_imgsize = np.array([[x * sx, y * sy, sc] for (x, y, sc) in peaks], dtype=np.float32)
        Sx = W0 / float(IMG_SIZE)
        Sy = H0 / float(IMG_SIZE)
        peaks_img = [(p[0] * Sx, p[1] * Sy, p[2]) for p in pred_px_imgsize]

        # Apply NMS in image space
        final_peaks = image_space_nms(peaks_img, radius_px=NMS_RADIUS, max_points=MAX_POINTS)

        # Save visualization
        vis = img_bgr.copy()
        coords_list = []
        for (x, y, sc) in final_peaks:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            coords_list.append([float(x), float(y)])

        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, vis)

        results_dict[fname] = coords_list
        print(f"[INFO] {fname}: {len(coords_list)} keypoints after NMS")

    # Save JSON
    json_out = os.path.join(OUTPUT_DIR, "keypoints.json")
    with open(json_out, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"\nAll keypoints saved in: {json_out}")

if __name__ == "__main__":
    main()
