#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate HeatmapNet on 5 random images from the test set.

- Loads 5 random images from test dir
- Runs inference with HeatmapNet
- Draws GT vs predicted points
- Computes MPE and PCK@8
"""

import os, json, random, sys
import numpy as np
import cv2
import torch

# === CONFIGURATION ===
TEST_DIR   = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Towel-scenes-custom/final_dataset/test"
ANN_FILE   = os.path.join(TEST_DIR, "dataset_coco.json")
WEIGHTS    = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/best_heatmap.pth"
IMG_SIZE   = 256
TOPK       = 4
THRESH     = 0.35
REFINE_WIN = 3
NUM_SAMPLES = 5
# ======================

try:
    from keypoint_heatmap_single_en import HeatmapNet
except Exception as e:
    print("[ERROR] Import HeatmapNet failed:", e)
    sys.exit(1)

# --- JSON loader ---
def load_json(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    img_by_id = {im["id"]: im for im in data["images"]}
    ann_by_img = {}
    for a in data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    return img_by_id, ann_by_img

# --- Model loader ---
def load_model(weights_path, device):
    model = HeatmapNet(out_ch=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

# --- Softargmax refine ---
def softargmax_refine(hm, x, y, win=3):
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

# --- Peak detector ---
def topk_peaks(hm, k=4, thresh=0.35, refine_win=3):
    H, W = hm.shape
    flat = hm.reshape(-1)
    idxs = np.argsort(-flat)
    peaks = []
    for idx in idxs:
        score = flat[idx]
        if score < thresh:
            break
        y, x = divmod(idx, W)
        rx, ry = softargmax_refine(hm, x, y, win=refine_win)
        peaks.append((rx, ry, float(score)))
        if len(peaks) == k:
            break
    return peaks

# --- Metrics ---
def greedy_mpe_and_pck(pred_pts, gt_pts, radius=8.0):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('nan'), 0.0
    used = set(); dists = []; correct = 0
    for p in pred_pts:
        d = np.linalg.norm(gt_pts - p[None, :], axis=1)
        j = int(np.argmin(d))
        if j not in used:
            used.add(j)
            dists.append(float(d[j]))
            if d[j] <= radius:
                correct += 1
    mpe = np.mean(dists) if dists else float('nan')
    pck = correct / len(gt_pts)
    return mpe, pck

# --- MAIN ---
def main():
    img_by_id, ann_by_img = load_json(ANN_FILE)
    sample_imgs = random.sample(list(img_by_id.values()), NUM_SAMPLES)

    device = torch.device("cpu")
    model = load_model(WEIGHTS, device)

    mpe_list, pck_list = [], []

    for im in sample_imgs:
        file_name = im["file_name"]
        W, H = int(im.get("width", 256)), int(im.get("height", 256))
        img_path = os.path.join(TEST_DIR, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize and preprocess
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.transpose(rgb, (2,0,1))).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            logits = model(t)
            hm = torch.sigmoid(logits)[0,0].cpu().numpy()

        # Predicted points
        peaks = topk_peaks(hm, k=TOPK, thresh=THRESH, refine_win=REFINE_WIN)
        h, w = hm.shape
        sx, sy = IMG_SIZE / float(w), IMG_SIZE / float(h)
        pred_px = np.array([[x*sx, y*sy] for x,y,_ in peaks], dtype=np.float32)

        # GT points
        anns = ann_by_img.get(im["id"], [])
        gt_vis = []
        if anns:
            kps = anns[0]["keypoints"]
            for i in range(0, 12, 3):
                x, y, v = kps[i], kps[i+1], kps[i+2]
                if v > 0:
                    x = x * (IMG_SIZE / W)
                    y = y * (IMG_SIZE / H)
                    gt_vis.append([x, y])
        gt_vis = np.array(gt_vis, dtype=np.float32)

        # Metrics
        mpe, pck = greedy_mpe_and_pck(pred_px, gt_vis, radius=8.0)
        mpe_list.append(mpe); pck_list.append(pck)
        print(f"{file_name}: MPE={mpe:.2f}px, PCK@8={pck:.2f}")

        # --- Visualization ---
        vis = resized.copy()
        # GT points = green
        for (x,y) in gt_vis:
            cv2.circle(vis, (int(x),int(y)), 5, (0,255,0), -1)
        # Pred points = red
        for (x,y) in pred_px:
            cv2.circle(vis, (int(x),int(y)), 5, (0,0,255), -1)
        cv2.imshow("GT (green) vs Predicted (red)", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC to stop early
            break

    print("\n=== Averages ===")
    print(f"MPE: {np.nanmean(mpe_list):.2f}px")
    print(f"PCK@8: {np.nanmean(pck_list):.2f}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
