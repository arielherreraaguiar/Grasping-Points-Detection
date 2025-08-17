#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time cloth corner keypoint detection with image-space NMS.

Key idea:
- Decode many peaks in heatmap space (don't be too strict there).
- Map peaks to original image coordinates.
- Apply Non-Maximum Suppression (NMS) in IMAGE space with a radius in pixels,
  keeping only the strongest peak per neighborhood.
- Display 1..4 points (not forced to show 4), no numeric labels.
- Optional video recording.

Hotkeys:
  q / ESC : quit
  [ / ]   : decrease / increase heatmap threshold
  - / +   : decrease / increase image-space NMS radius (in pixels)
"""

import os, sys, time, cv2, numpy as np, torch

# ===== USER CONFIG =====
WEIGHTS_PATH = "/home/ariel/Downloads/Thesis_CV/Grasping-Points-Detection/best_heatmap.pth"
IMG_SIZE     = 256          # model input size (training size)
USE_REALSENSE = True        # try RealSense first; set False to force webcam
CAMERA_INDEX = 0            # fallback webcam index
SHOW_DEPTH   = False        # show RealSense depth next to points if available

# Heatmap → candidates
TOPK_HEAT   = 50            # take many candidates from the heatmap
HM_THRESH   = 0.9         # initial (low) threshold for heatmap peaks

# Image-space NMS (postprocess AFTER mapping to original image)
NMS_RADIUS_PX = 22          # initial NMS radius in pixels (image space)
MAX_POINTS     = 4          # cap results to at most 4 points

# Visualization / Video
DRAW_RADIUS = 5
DRAW_COLOR  = (0, 0, 255)   # red
SAVE_VIDEO  = True
VIDEO_PATH  = "realtime_keypoints.mp4"
VIDEO_FPS   = 30
# ========================

# Import model and helpers
try:
    from keypoint_heatmap_single_en import HeatmapNet, order_tl_tr_br_bl
except Exception as e:
    print("[ERROR] Could not import HeatmapNet/order_tl_tr_br_bl:",
          e, "\nMake sure keypoint_heatmap_single_en.py is in the SAME folder.")
    sys.exit(1)

# Optional RealSense
try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except Exception:
    RS_AVAILABLE = False

def load_model(weights_path: str, device: torch.device):
    """Load HeatmapNet with weights; accepts pure state_dict or wrapped checkpoints."""
    model = HeatmapNet(out_ch=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def letterbox_bgr(img_bgr, new_size=256, pad_val=114):
    """Resize with aspect ratio and pad to new_size×new_size; return canvas and undo-info."""
    h, w = img_bgr.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), pad_val, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    info = dict(scale=scale, top=top, left=left, new_h=nh, new_w=nw, orig_h=h, orig_w=w)
    return canvas, info

def inv_letterbox_point(x_hat, y_hat, info, heat_w, heat_h):
    """
    Map (x_hat, y_hat) from heatmap coordinates to original image pixels.
    heatmap → model input (256x256) → remove pad → unscale to original.
    """
    sx = IMG_SIZE / float(heat_w)
    sy = IMG_SIZE / float(heat_h)
    x_in = x_hat * sx
    y_in = y_hat * sy
    x_no_pad = x_in - info["left"]
    y_no_pad = y_in - info["top"]
    if info["scale"] > 0:
        x_orig = x_no_pad / info["scale"]
        y_orig = y_no_pad / info["scale"]
    else:
        x_orig, y_orig = x_in, y_in
    x_orig = float(np.clip(x_orig, 0, info["orig_w"] - 1))
    y_orig = float(np.clip(y_orig, 0, info["orig_h"] - 1))
    return x_orig, y_orig

def softargmax_refine(hm, x, y, win=3):
    """Refine integer peak (x,y) using local soft-argmax (centroid) in a 3×3 or 5×5 window."""
    H, W = hm.shape
    x, y = int(x), int(y)
    x0, x1 = max(0, x - win//2), min(W, x + win//2 + 1)
    y0, y1 = max(0, y - win//2), min(H, y + win//2 + 1)
    patch = hm[y0:y1, x0:x1]
    if patch.size == 0:
        return float(x), float(y)
    wsum = patch.sum()
    if wsum <= 1e-8:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rx = (patch * xx).sum() / wsum
    ry = (patch * yy).sum() / wsum
    return float(rx), float(ry)

def decode_many_heatmap_peaks(hm, K=50, thresh=0.12, refine=True):
    """
    Decode MANY peaks from a heatmap (numpy [H,W] in [0,1]).
    Returns list of (x, y, score) in heatmap coords (not image).
    """
    H, W = hm.shape
    flat = hm.reshape(-1)
    idxs = np.argsort(-flat)  # descending
    peaks = []
    used = np.zeros_like(flat, dtype=bool)  # (optional) duplicate suppression on identical idx
    for idx in idxs:
        score = flat[idx]
        if score < thresh:
            break
        if used[idx]:
            continue
        y, x = divmod(idx, W)
        if refine:
            rx, ry = softargmax_refine(hm, x, y, win=3)
            peaks.append((rx, ry, float(score)))
        else:
            peaks.append((float(x), float(y), float(score)))
        used[idx] = True
        if len(peaks) >= K:
            break
    return peaks

def image_space_nms(peaks_img, radius_px=22, max_points=4):
    """
    Greedy NMS in IMAGE space. peaks_img: list of (x, y, score) in *original image* pixels.
    Keeps the highest-score peak, suppresses other peaks within 'radius_px' of any kept peak.
    Returns up to 'max_points' peaks.
    """
    if len(peaks_img) == 0:
        return []
    # sort by score DESC
    peaks = sorted(peaks_img, key=lambda p: p[2], reverse=True)
    kept = []
    for x, y, sc in peaks:
        suppress = False
        for kx, ky, ksc in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 <= radius_px ** 2:
                suppress = True
                break
        if not suppress:
            kept.append((x, y, sc))
        if len(kept) >= max_points:
            break
    return kept

def start_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align

def main():
    global HM_THRESH, NMS_RADIUS_PX
    device = torch.device("cpu")  # CPU by default (change to 'cuda' if you wish)

    if not os.path.isfile(WEIGHTS_PATH):
        print("[ERROR] Weights not found:", WEIGHTS_PATH); sys.exit(1)
    model = load_model(WEIGHTS_PATH, device)
    print("[INFO] Model loaded:", WEIGHTS_PATH)

    # Video source
    use_rs = USE_REALSENSE and RS_AVAILABLE
    pipeline = align = None
    cap = None
    if use_rs:
        try:
            pipeline, align = start_realsense()
            print("[INFO] RealSense started.")
        except Exception as e:
            print("[WARN] RealSense failed, fallback to webcam:", e)
            use_rs = False
    if not use_rs:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[ERROR] Webcam could not be opened."); sys.exit(1)
        print("[INFO] Webcam started.")

    # Prepare video writer (uses native frame size, not 256×256)
    writer = None
    frame_w = frame_h = None

    prev = time.time()
    while True:
        # ---- capture ----
        if use_rs:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_bgr = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
        else:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            depth_frame = None

        if frame_w is None:
            frame_h, frame_w = frame_bgr.shape[:2]
            if SAVE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (frame_w, frame_h))

        # ---- preprocess for model (letterbox to 256x256) ----
        lb, info = letterbox_bgr(frame_bgr, new_size=IMG_SIZE, pad_val=114)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float().to(device)

        # ---- inference ----
        with torch.no_grad():
            logits = model(t)                 # [1,1,64,64] if stride=4
            hm = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # ---- decode many peaks (heatmap space) ----
        peaks_hm = decode_many_heatmap_peaks(hm, K=TOPK_HEAT, thresh=HM_THRESH, refine=True)

        # ---- map peaks to original image coordinates ----
        Hh, Wh = hm.shape  # e.g., 64×64
        peaks_img = []
        for xh, yh, sc in peaks_hm:
            x_orig, y_orig = inv_letterbox_point(xh, yh, info, Wh, Hh)
            peaks_img.append((x_orig, y_orig, sc))

        # ---- image-space NMS (keep best within radius) ----
        kpts_xy = image_space_nms(peaks_img, radius_px=NMS_RADIUS_PX, max_points=MAX_POINTS)

        # Optional ordering if you *did* get 4 points
        if len(kpts_xy) == 4:
            ordered = order_tl_tr_br_bl(np.array([(x, y) for x, y, _ in kpts_xy]))
            # keep original scores by nearest matching (not strictly necessary for drawing)
            kpts_xy = [(float(p[0]), float(p[1]), kpts_xy[i][2]) for i, p in enumerate(ordered)]

        # ---- draw ----
        vis = frame_bgr.copy()
        for (x, y, sc) in kpts_xy:
            cv2.circle(vis, (int(x), int(y)), DRAW_RADIUS, DRAW_COLOR, -1, cv2.LINE_AA)
            if SHOW_DEPTH and use_rs and depth_frame is not None:
                d = depth_frame.get_distance(int(x), int(y))
                cv2.putText(vis, f"{d:.3f} m", (int(x) + 6, int(y) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # status overlay
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev)); prev = now
        cv2.putText(vis, f"FPS: {fps:.1f}  HM_THR: {HM_THRESH:.2f}  NMS_R: {NMS_RADIUS_PX}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # show & write
        cv2.imshow("Cloth Corner Keypoints (Image-space NMS)", vis)
        if SAVE_VIDEO and writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('['):
            HM_THRESH = max(0.01, HM_THRESH - 0.02)   # more strict
        elif key == ord(']'):
            HM_THRESH = min(0.95, HM_THRESH + 0.02)   # more permissive
        elif key == ord('-'):
            NMS_RADIUS_PX = max(5, NMS_RADIUS_PX - 2)
        elif key == ord('+') or key == ord('='):
            NMS_RADIUS_PX = min(200, NMS_RADIUS_PX + 2)

    # cleanup
    if SAVE_VIDEO and writer is not None:
        writer.release()
        print(f"[INFO] Video saved to: {VIDEO_PATH}")
    if USE_REALSENSE and RS_AVAILABLE and pipeline is not None:
        pipeline.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
