#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Segmentation with Intel RealSense D435 + YOLOv8

This script performs real-time instance segmentation (or detection fallback) using a YOLOv8 model
on RGB frames from an Intel RealSense D435. Depth is aligned to color, and per-object distances
are estimated from the depth map.

Notes
-----
- The model used here can be your custom YOLOv8 **segmentation** model trained on a **Roboflow** dataset.
- Training notebook (Colab): https://colab.research.google.com/drive/1Kn5kF8jvKpI50eObxHSCjKAVuxXxTHC1?usp=sharing
- Static-image inference notebook (Colab) for wrinkle detection/segmentation:
  https://colab.research.google.com/drive/12kBpE8gECRDdIp1zmT7XGCZ3YESo1MEW?usp=sharing

Quick Start
-----------
  python realsense_yolov8_seg.py --weights yolov8n-seg.pt
  python realsense_yolov8_seg.py --weights best.pt --conf 0.4 --imgsz 640 --device cuda:0
  python realsense_yolov8_seg.py --weights best.pt --show-depth --save out.mp4

Dependencies
------------
  pip install ultralytics==8.2.0 pyrealsense2 opencv-python torch torchvision
  # Make sure Intel RealSense SDK (librealsense) is installed and the D435 is connected

How it works (high level)
-------------------------
1) Initialize RealSense streams for color and depth at 640x480 @ 30 FPS.
2) Align depth to color so each RGB pixel has a corresponding depth value.
3) Run the YOLOv8 model on each RGB frame (segmentation preferred).
4) For each predicted mask:
     - Overlay mask with transparency on the RGB frame.
     - Draw a bounding box and label with class, confidence, and median distance.
5) If masks are not available (e.g., detection-only model), draw boxes and compute distance
   from the box center.
6) Display FPS and optionally save the visualization to a video file.
"""

import argparse
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

# Intel RealSense SDK
import pyrealsense2 as rs

# YOLOv8 (Ultralytics)
from ultralytics import YOLO
import torch


@dataclass
class AppConfig:
    """Runtime configuration parsed from CLI."""
    weights: str
    conf: float = 0.5
    imgsz: int = 640
    device: str = ""        # "", "cpu" or "cuda:0"
    save: Optional[str] = None
    show_depth: bool = False
    max_distance_m: float = 5.0  # clamp depth stats (meters)
    alpha: float = 0.45           # mask overlay transparency
    thickness: int = 2            # bbox line thickness
    font_scale: float = 0.5
    line_height: int = 18


def parse_args() -> AppConfig:
    """Parse command-line arguments and return an AppConfig."""
    p = argparse.ArgumentParser(description="Real-time segmentation with Intel RealSense D435 + YOLOv8")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 weights (.pt or .onnx)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (square)")
    p.add_argument("--device", type=str, default="", help='Device: "" (auto), "cpu", or "cuda:0"')
    p.add_argument("--save", type=str, default=None, help="Optional path to save output video (e.g., out.mp4)")
    p.add_argument("--show-depth", action="store_true", help="Also show a depth colormap window")
    p.add_argument("--max-distance-m", type=float, default=5.0, help="Clamp depth stats to this max distance (m)")
    p.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha [0..1]")
    args = p.parse_args()

    return AppConfig(
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show_depth=args.show_depth,
        max_distance_m=args.max_distance_m,
        alpha=args.alpha,
    )


def pick_device(pref: str) -> str:
    """Return a valid torch device string, preferring CUDA if available unless overridden."""
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def init_model(cfg: AppConfig) -> YOLO:
    """Load a YOLOv8 model from provided weights. Works with .pt or .onnx via Ultralytics."""
    model = YOLO(cfg.weights)
    # Fuse layers for a small inference speed-up (only affects PyTorch models).
    try:
        model.fuse()
    except Exception:
        pass
    return model


def init_realsense() -> Tuple[rs.pipeline, rs.align, rs.pipeline_profile]:
    """
    Initialize the RealSense pipeline for color + depth, aligned to color.
    Returns (pipeline, align, profile).
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color + depth at 640x480 30 FPS. You can increase resolution if you have GPU headroom.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Align depth to color: each RGB pixel gets a matching depth value after alignment.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Print the depth scale (meters per unit). Depth frame values are uint16 "units".
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] RealSense depth scale: {depth_scale:.6f} meters per unit")

    # Try enabling auto-exposure on the color sensor (if supported).
    try:
        sensors = profile.get_device().query_sensors()
        color_sensor = next((s for s in sensors if s.is_color_sensor()), None)
        if color_sensor and color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    except Exception:
        pass

    return pipeline, align, profile


def meter_stats_from_mask(depth_m: np.ndarray, mask: np.ndarray, clamp_max_m: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute median and 10th percentile depth (in meters) from pixels under a binary mask.
    Returns (median_m, p10_m). If invalid or empty, returns (None, None).

    Rationale:
      - Median is robust against outliers (holes/noise in depth).
      - 10th percentile is a proxy for the nearest surface.
    """
    if depth_m is None or depth_m.size == 0:
        return None, None

    m = mask.astype(bool)
    if not np.any(m):
        return None, None

    vals = depth_m[m]
    vals = vals[(vals > 0) & np.isfinite(vals)]
    if vals.size == 0:
        return None, None

    vals = np.clip(vals, 0.0, clamp_max_m)
    med = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    return med, p10


def overlay_mask(img: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.45) -> None:
    """
    In-place overlay of a single binary mask with transparency on a BGR image.
    The mask should be uint8 0/255 or boolean; if not, it will be thresholded at 0.5.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    overlay = img.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color_bgr, dtype=np.float32) * alpha).astype(np.uint8)
    img[:] = overlay


def random_color(i: int) -> Tuple[int, int, int]:
    """Deterministic pseudo-random BGR color for visualization based on index."""
    rng = np.random.default_rng(i * 12345 + 7)
    return tuple(int(x) for x in rng.integers(64, 256, size=3))  # avoid very dark colors


def draw_label(img: np.ndarray, xy: Tuple[int, int], text: str, font_scale: float = 0.5) -> None:
    """Draw a readable text label with a dark background box."""
    x, y = int(xy[0]), int(xy[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, 1)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    cfg = parse_args()
    device = pick_device(cfg.device)
    print(f"[INFO] Using device: {device}")

    # Load model and move to device
    model = init_model(cfg)
    model.to(device)
    print("[INFO] Model loaded. Class names:", model.names)

    # Initialize RealSense
    pipeline, align, profile = init_realsense()

    # Prepare video writer if requested
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while True:
            t0 = time.time()

            # Wait for a new set of frames and align depth to color
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color = np.asanyarray(color_frame.get_data())  # BGR uint8 image
            depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 depth (units)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_m = depth_raw * depth_scale  # convert to meters

            # YOLOv8 inference
            # Ultralytics will infer if the weights are segmentation or detection
            results = model.predict(color, imgsz=cfg.imgsz, conf=cfg.conf, device=device, verbose=False)

            vis = color.copy()

            # Iterate over each result in the batch (here batch size is 1 from the camera)
            for r in results:
                # If we have segmentation masks
                if getattr(r, "masks", None) is not None and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()           # (N, H, W) in [0..1]
                    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
                    clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else None
                    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else None

                    for i, m in enumerate(masks):
                        color_i = random_color(i)

                        # Resize mask to frame size if needed
                        if m.shape != vis.shape[:2]:
                            m = cv2.resize(m, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # Visual overlay
                        overlay_mask(vis, (m > 0.5).astype(np.uint8) * 255, color_i, alpha=cfg.alpha)

                        # Depth statistics from the masked region
                        med, p10 = meter_stats_from_mask(depth_m, m > 0.5, cfg.max_distance_m)

                        # Build the label
                        label = ""
                        if clss is not None and i < len(clss):
                            cname = model.names.get(int(clss[i]), str(int(clss[i])))
                            conf = float(confs[i]) if confs is not None else 0.0
                            if med is not None:
                                label = f"{cname} {conf:.2f} | med {med:.2f} m"
                            else:
                                label = f"{cname} {conf:.2f}"
                        else:
                            if med is not None:
                                label = f"med {med:.2f} m"

                        # Draw bbox and label if boxes exist
                        if boxes is not None and i < len(boxes):
                            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
                            cv2.rectangle(vis, (x1, y1), (x2, y2), color_i, cfg.thickness)
                            draw_label(vis, (x1, y1), label, cfg.font_scale)
                        else:
                            # Fallback: top-left corner stacked labels
                            draw_label(vis, (10, 20 + i * cfg.line_height), label, cfg.font_scale)

                # No masks? Fallback to detection-only visualization
                elif r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()

                    for i, (xyxy, cls_id, conf) in enumerate(zip(boxes, clss, confs)):
                        color_i = random_color(i)
                        x1, y1, x2, y2 = xyxy.astype(int).tolist()
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color_i, cfg.thickness)

                        # Estimate depth from the center pixel of the box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        d = float(depth_m[cy, cx]) if 0 <= cy < depth_m.shape[0] and 0 <= cx < depth_m.shape[1] else float("nan")
                        d_txt = f"{d:.2f} m" if math.isfinite(d) and d > 0 else "n/a"

                        cname = model.names.get(int(cls_id), str(int(cls_id)))
                        label = f"{cname} {conf:.2f} | {d_txt}"
                        draw_label(vis, (x1, y1), label, cfg.font_scale)

            # FPS overlay
            fps = 1.0 / max(1e-6, time.time() - t0)
            draw_label(vis, (10, vis.shape[0] - 10), f"{fps:.1f} FPS | {device}", cfg.font_scale)

            # Initialize writer lazily when we know frame size
            if writer is None and cfg.save is not None:
                writer = cv2.VideoWriter(cfg.save, fourcc, 30, (vis.shape[1], vis.shape[0]))
            if writer is not None:
                writer.write(vis)

            # Show visualization
            cv2.imshow("YOLOv8 + RealSense (RGB)", vis)

            if cfg.show_depth:
                # Depth colormap for display only (clamped to max_distance_m meters)
                depth_show = np.clip(depth_m, 0, cfg.max_distance_m)
                depth_show = (depth_show / cfg.max_distance_m * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)
                cv2.imshow("Depth (m, colormap)", depth_color)

            # Quit on 'q' or 'Esc'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
