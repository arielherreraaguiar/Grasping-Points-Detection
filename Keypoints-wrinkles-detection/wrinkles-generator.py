#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrinkle detection and line extraction script.
- Loads YOLOv8 model (ultralytics)
- Processes all images from "images/"
- Saves masks in "masks/"
- Outputs a JSON file with line start/end coordinates per image
"""

import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# ================= USER CONFIG =================
MODEL_PATH = "best.pt"     # Ruta a tu modelo YOLO
IMAGES_DIR = "images"      # Carpeta de entrada
MASKS_DIR  = "masks"       # Carpeta de salida
CONF_THRES = 0.15
IMGSZ      = 640
# ===============================================

os.makedirs(MASKS_DIR, exist_ok=True)

# ---------- Load YOLO model ----------
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

results_json = {}

# ---------- Process each image ----------
for fname in sorted(os.listdir(IMAGES_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_DIR, fname)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"[WARN] Could not read {fname}, skipping.")
        continue
    h, w = orig_img.shape[:2]

    # Run YOLO inference
    results = model.predict(source=img_path, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)

    # Initialize best contour tracking
    best_contour = None
    best_length = 0
    best_box = None

    # Loop through detections
    for box in results[0].boxes.xyxy.cpu().numpy():  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)

        roi = orig_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 1: Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Step 2: Morph ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        closed = cv2.dilate(closed, kernel, iterations=1)

        # Step 3: Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Longest contour
        for cnt in contours:
            if len(cnt) >= 5:
                (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
                length = max(MA, ma)
            else:
                rect = cv2.minAreaRect(cnt)
                (rw, rh) = rect[1]
                length = max(rw, rh)

            if length > best_length:
                best_length = length
                best_contour = cnt
                best_box = (x1, y1, x2, y2)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    if best_contour is not None and best_box is not None:
        x1, y1, x2, y2 = best_box
        shifted_contour = best_contour + [x1, y1]
        cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=-1)

    # Save mask
    mask_name = os.path.splitext(fname)[0] + "_mask.png"
    mask_path = os.path.join(MASKS_DIR, mask_name)
    cv2.imwrite(mask_path, mask)
    print(f"[INFO] Saved mask: {mask_path}")

    # Extract line endpoints from mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        results_json[fname] = {"start": None, "end": None}
        continue

    points = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(points)

    dist_matrix = distance.cdist(hull[:, 0, :], hull[:, 0, :], "euclidean")
    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    start_point, end_point = tuple(hull[i][0]), tuple(hull[j][0])

    # Save result
    results_json[fname] = {
        "start": [int(start_point[0]), int(start_point[1])],
        "end": [int(end_point[0]), int(end_point[1])]
    }

# ---------- Save JSON ----------
json_path = os.path.join(MASKS_DIR, "lines.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=4)
print(f"✅ JSON with line coordinates saved to: {json_path}")
