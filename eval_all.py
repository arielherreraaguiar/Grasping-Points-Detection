#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import math
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================= CONFIG =================
TEST_DIR    = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/Towel-scenes-custom/final_dataset/test"
ANN_FILE    = os.path.join(TEST_DIR, "dataset_coco.json")

OWN_WEIGHTS = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/best_heatmap.pth"
LIPS_CKPT   = "model_Ghent.ckpt"

IMG_SIZE    = 256
OUT_PLOT    = "compare_mpe_boxplot.png"
OUT_CSV     = "mpe_results.csv"

OWN_TOPK       = 4
OWN_THRESH     = 0.35
OWN_REFINE_WIN = 3
# ==========================================

try:
    from keypoint_heatmap_single_en import HeatmapNet
except Exception as e:
    print("[ERROR] Could not import HeatmapNet:", e)
    sys.exit(1)

from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap


def load_json(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    img_by_id = {im["id"]: im for im in data["images"]}
    ann_by_img = {}
    for a in data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    return img_by_id, ann_by_img

def greedy_mpe(pred_pts_xy, gt_pts_xy):
    if len(pred_pts_xy) == 0 or len(gt_pts_xy) == 0:
        return float('nan')
    used = set()
    dists = []
    for p in pred_pts_xy:
        d = np.linalg.norm(gt_pts_xy - p[None, :], axis=1)
        j = int(np.argmin(d))
        if j not in used:
            used.add(j)
            dists.append(float(d[j]))
    return np.mean(dists) if dists else float('nan')

def nanmean_std(arr):
    clean = [x for x in arr if not math.isnan(x)]
    if len(clean) == 0:
        return float('nan'), float('nan')
    return float(np.mean(clean)), float(np.std(clean))


def softargmax_refine(hm, x, y, win=3):
    H, W = hm.shape
    x, y = int(x), int(y)
    x0, x1 = max(0, x - win//2), min(W, x + win//2 + 1)
    y0, y1 = max(0, y - win//2), min(H, y + win//2 + 1)
    patch = hm[y0:y1, x0:x1]
    if patch.size == 0: return float(x), float(y)
    s = patch.sum()
    if s <= 1e-8: return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rx = (patch * xx).sum() / s
    ry = (patch * yy).sum() / s
    return float(rx), float(ry)

def topk_peaks(hm, k=4, thresh=0.35, refine_win=3):
    H, W = hm.shape
    flat = hm.reshape(-1)
    idxs = np.argsort(-flat)
    peaks, taken = [], np.zeros_like(hm, dtype=bool)
    for idx in idxs:
        score = float(flat[idx])
        if score < thresh: break
        y, x = divmod(idx, W)
        if taken[y, x]: continue
        rx, ry = softargmax_refine(hm, x, y, win=refine_win)
        peaks.append((rx, ry, score))
        r = max(1, refine_win)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        taken[y0:y1, x0:x1] = True
        if len(peaks) == k: break
    return peaks


def load_own_model(device):
    model = HeatmapNet(out_ch=1).to(device)
    ckpt = torch.load(OWN_WEIGHTS, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def load_lips_model(device):
    model = KeypointDetector.load_from_checkpoint(
        LIPS_CKPT, map_location=device, backbone_type="Unet"
    )
    model.eval()
    return model


def predict_own(model, img_bgr):
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(t)
        hm = torch.sigmoid(logits)[0, 0].cpu().numpy()
    Hh, Wh = hm.shape
    peaks = topk_peaks(hm, k=OWN_TOPK, thresh=OWN_THRESH, refine_win=OWN_REFINE_WIN)
    sx, sy = IMG_SIZE / float(Wh), IMG_SIZE / float(Hh)
    pts = np.array([[x * sx, y * sy] for x, y, _ in peaks], dtype=np.float32)
    return pts

def predict_lips(model, pil_img):
    to_tensor = T.ToTensor()
    resize_transform = T.Resize((IMG_SIZE, IMG_SIZE))
    img_tensor = to_tensor(resize_transform(pil_img)).unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(img_tensor)
    # mantener como tensor
    heatmap = heatmaps[0, 0]
    kps = get_keypoints_from_heatmap(heatmap, 25, 4)
    if len(kps) == 0:
        return np.empty((0, 2), dtype=np.float32)
    Hh, Wh = heatmap.shape
    sx, sy = IMG_SIZE / float(Wh), IMG_SIZE / float(Hh)
    pts = np.array([[kp[0] * sx, kp[1] * sy] for kp in kps], dtype=np.float32)
    return pts


def main():
    device = torch.device("cpu")

    img_by_id, ann_by_img = load_json(ANN_FILE)
    all_images = list(img_by_id.values())
    own_model  = load_own_model(device)
    lips_model = load_lips_model(device)

    mpe_own, mpe_lips = [], []

    with open(OUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "mpe_own", "mpe_lips"])

        for im in all_images:
            file_name = im["file_name"]
            W, H = int(im.get("width", 256)), int(im.get("height", 256))
            img_path = os.path.join(TEST_DIR, file_name)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            pil_img = Image.open(img_path).convert("RGB")

            gt_vis = []
            anns = ann_by_img.get(im["id"], [])
            if anns:
                kps = anns[0]["keypoints"]
                for i in range(0, len(kps), 3):
                    x, y, v = kps[i], kps[i+1], kps[i+2]
                    if v > 0:
                        x = x * (IMG_SIZE / W)
                        y = y * (IMG_SIZE / H)
                        gt_vis.append([x, y])
            gt_vis = np.array(gt_vis, dtype=np.float32)

            pred_own  = predict_own(own_model, img_bgr)
            pred_lips = predict_lips(lips_model, pil_img)

            mpe_i_own  = greedy_mpe(pred_own,  gt_vis)
            mpe_i_lips = greedy_mpe(pred_lips, gt_vis)

            mpe_own.append(mpe_i_own)
            mpe_lips.append(mpe_i_lips)

            print(f"{file_name}: MPE Own={mpe_i_own}, MPE Lips={mpe_i_lips}")
            writer.writerow([file_name, mpe_i_own, mpe_i_lips])

    mean_own,  std_own  = nanmean_std(mpe_own)
    mean_lips, std_lips = nanmean_std(mpe_lips)

    print("\n=== Results (NaN-safe) ===")
    print(f"Own Model  -> MPE: {mean_own:.4f} ± {std_own:.4f} px")
    print(f"Lips Model -> MPE: {mean_lips:.4f} ± {std_lips:.4f} px")

    plt.figure(figsize=(7, 7))
    plt.boxplot([mpe_own, mpe_lips],
                labels=["MPE Own Model", "MPE Lips et al. Model"],
                showfliers=False, whis=[0, 100])
    plt.ylabel("Error [pixels]")
    plt.title("Comparison of MPE between Models")
    plt.grid(True, linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()
    print(f"Saved boxplot -> {os.path.abspath(OUT_PLOT)}")
    print(f"Saved CSV -> {os.path.abspath(OUT_CSV)}")


if __name__ == "__main__":
    main()
