#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# ================= USER CONFIG =================
MODEL_PATH = "best-new.pt"   
BASE_DIR = "images"          
CONF_THRES = 0.15
IMGSZ = 640
# ===============================================

def process_wrinkle_roi(img, box):
    """
    Analyzes the Region of Interest (ROI) defined by the box to find the wrinkle contour.
    Logic extracted from detector_node.py.
    
    Args:
        img: The original image (numpy array).
        box: The bounding box [x1, y1, x2, y2].
        
    Returns:
        tuple: (length, start_point_global, end_point_global, contour_global)
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Validate crop bounds
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        return 0, None, None, None
        
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return 0, None, None, None

    # 1. Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    # 3. Morphological Operations (Close gaps to form a blob)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    closed = cv2.dilate(closed, kernel, iterations=1)

    # 4. Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, None, None, None

    # 5. Find the longest contour in this specific box
    best_cnt_local = None
    max_len = 0
    
    for cnt in contours:
        length = 0
        # Calculate length based on shape (Ellipse fit or MinAreaRect)
        if len(cnt) >= 5:
            try:
                (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
                length = max(MA, ma)
            except:
                pass
        else:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            length = max(rw, rh)
        
        if length > max_len:
            max_len = length
            best_cnt_local = cnt

    if best_cnt_local is None:
        return 0, None, None, None

    # 6. Calculate Start/End points using Convex Hull and Euclidean Distance
    roi_h, roi_w = gray.shape
    mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.drawContours(mask_roi, [best_cnt_local], -1, 255, thickness=-1)
    
    ys, xs = np.where(mask_roi > 0)
    if len(xs) == 0: return 0, None, None, None
    
    points = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(points)
    
    if len(hull) < 2: return 0, None, None, None
    
    hull_pts = hull[:, 0, :] 
    dist_matrix = distance.cdist(hull_pts, hull_pts, "euclidean")
    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    
    p_start_local = hull_pts[i]
    p_end_local = hull_pts[j]
    
    # 7. Convert local coordinates back to global image coordinates
    p_start_global = (p_start_local[0] + x1, p_start_local[1] + y1)
    p_end_global = (p_end_local[0] + x1, p_end_local[1] + y1)
    
    best_cnt_global = best_cnt_local + [x1, y1]
    
    return max_len, p_start_global, p_end_global, best_cnt_global

def main():
    print(f"Loading YOLO model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model {MODEL_PATH}. Check if the file exists.")
        return

    # Check base directory
    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] Directory '{BASE_DIR}' not found. Please run scene-generator.py first.")
        return

    # Find scene folders
    scene_folders = sorted([f for f in os.listdir(BASE_DIR) if f.startswith("scene")])
    if not scene_folders:
        print(f"[WARNING] No scene folders found in '{BASE_DIR}'.")

    for scene_name in scene_folders:
        scene_path = os.path.join(BASE_DIR, scene_name)
        raw_images_path = os.path.join(scene_path, "raw_images")
        
        # Define output directories
        boxes_dir = os.path.join(scene_path, "boxes")
        seg_dir = os.path.join(scene_path, "segmentation")
        os.makedirs(boxes_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"Processing wrinkles for {scene_name}...")
        
        if not os.path.exists(raw_images_path):
            print(f"[WARN] No raw_images folder in {scene_name}, skipping.")
            continue

        for fname in sorted(os.listdir(raw_images_path)):
            if not fname.endswith(("_raw.png", ".png", ".jpg")):
                continue
            
            img_path = os.path.join(raw_images_path, fname)
            orig_img = cv2.imread(img_path)
            if orig_img is None: continue

            # --- Prediction ---
            results = model.predict(orig_img, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)

            # ---------------------------------------------------------
            # TASK 1: BOXES FOLDER
            # Save raw image with ALL detected boxes drawn (using ultralytics plot)
            # ---------------------------------------------------------
            # .plot() returns the BGR numpy array with boxes and labels
            img_boxes = results[0].plot() 
            boxes_filename = fname.replace("_raw.png", "_boxes.png")
            cv2.imwrite(os.path.join(boxes_dir, boxes_filename), img_boxes)

            # ---------------------------------------------------------
            # TASK 2: SEGMENTATION FOLDER
            # Draw ONLY the segmentation of the LONGEST wrinkle + Blue Line
            # ---------------------------------------------------------
            global_best_len = 0
            global_start = None
            global_end = None
            global_contour = None
            
            # 1. Iterate through all boxes to find the single longest wrinkle in the image
            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    length, p_start, p_end, cnt = process_wrinkle_roi(orig_img, xyxy)
                    
                    if length > global_best_len:
                        global_best_len = length
                        global_start = p_start
                        global_end = p_end
                        global_contour = cnt
            
            # 2. Create the visualization on the raw image
            img_seg = orig_img.copy()

            if global_start and global_end:
                # Draw the contour filled in WHITE (as requested)
                # -1 thickness means fill
                cv2.drawContours(img_seg, [global_contour], -1, (255, 255, 255), -1) 
                
                # Draw the line connecting farthest points in BLUE (BGR: 255, 0, 0)
                cv2.line(img_seg, global_start, global_end, (255, 0, 0), 3)
                
                # Draw endpoints (Red circles for contrast, or Blue if preferred)
                cv2.circle(img_seg, global_start, 5, (0, 0, 255), -1)
                cv2.circle(img_seg, global_end, 5, (0, 0, 255), -1)
                
                # Add Label "Longest wrinkle"
                cv2.putText(img_seg, "Longest wrinkle", (global_start[0], global_start[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            seg_filename = fname.replace("_raw.png", "_seg.png")
            cv2.imwrite(os.path.join(seg_dir, seg_filename), img_seg)
            
    print("Wrinkle processing completed.")

if __name__ == "__main__":
    main()