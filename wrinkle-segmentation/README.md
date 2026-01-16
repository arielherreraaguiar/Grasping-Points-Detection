# RealSense + YOLOv8 (Segmentation) — Real-Time RGB‑D Wrinkle Detection

This repository shows a **complete workflow** to train a segmentation model on **Roboflow**, test it on **static images**, and finally run **real-time segmentation** using an **Intel RealSense D435** (RGB‑D). The goal is to segment wrinkles (or your custom classes) and **estimate distances** using the depth map.

## 1) Train your model on Roboflow (Colab)
- We trained the YOLOv8 **segmentation** model on a dataset hosted in **Roboflow**.
- Use this Colab to train (adjust classes, epochs, and hyperparameters as needed):
  - **Training Notebook:** https://colab.research.google.com/drive/1Kn5kF8jvKpI50eObxHSCjKAVuxXxTHC1?usp=sharing

> Export your best weights as `best.pt` (or export to ONNX if preferred).

## 2) Static-image inference (Colab)
- Before going real-time, validate results on images (wrinkle segmentation/detection):
  - **Inference Notebook (Static Images):** https://colab.research.google.com/drive/12kBpE8gECRDdIp1zmT7XGCZ3YESo1MEW?usp=sharing

This notebook helps confirm your classes, masks, and confidence thresholds.

## 3) Real-time segmentation with Intel RealSense D435
Use the included script to run segmentation live on RGB frames and compute distances from the aligned depth map.

### Installation
```bash
python -m venv .venv && source .venv/bin/activate            # optional (Linux/macOS)
pip install ultralytics==8.2.0 pyrealsense2 opencv-python torch torchvision
# Make sure the Intel RealSense SDK (librealsense) is installed and your D435 is connected
```

### Run
```bash
python realsense_yolov8_seg.py --weights yolov8n-seg.pt --conf 0.4 --imgsz 640
# or, with your trained weights from Roboflow training:
python realsense_yolov8_seg.py --weights best.pt --device cuda:0 --show-depth --save out.mp4
```

### What the script does (Step-by-step)
1. **Camera setup**  
   Initializes **color** and **depth** streams at 640×480@30 FPS and **aligns depth to color** so each RGB pixel has a matching depth value.
2. **Model inference**  
   Runs **YOLOv8** on RGB frames. If your weights are **segmentation**, it retrieves **instance masks**. If they are detection-only, it falls back to **bounding boxes**.
3. **Visualization**  
   - Overlays each predicted **mask** with semi-transparency.
   - Draws the **bounding box**, **class**, and **confidence**.
4. **Depth-based distance**  
   - Computes the **median distance (m)** within each mask (robust to noise/outliers).  
   - Also computes the **10th percentile** (a proxy for the nearest visible surface).
5. **Performance and output**  
   - Displays **FPS** and the selected **device** (CPU/CUDA).  
   - Optionally **saves** the annotated video (`--save out.mp4`).  
   - Optional **depth colormap** view (`--show-depth`).

### Why depth alignment matters
RealSense depth is aligned so that for each pixel `(u, v)` in RGB you can read a reliable metric distance `D(u,v)` in meters. This enables per-object distance statistics using the predicted masks.

### Tips
- Prefer a **segmentation** model (e.g., `yolov8n-seg.pt`) trained on your **Roboflow** dataset for best mask quality.
- If you face pickling issues with certain PyTorch versions, export to **ONNX**:
  ```bash
  yolo export model=best.pt format=onnx opset=12
  python realsense_yolov8_seg.py --weights best.onnx
  ```
- Use GPU if available:
  ```bash
  python realsense_yolov8_seg.py --weights best.pt --device cuda:0
  ```

### Hotkeys
- `q` or `Esc`: Quit the app

### Troubleshooting
- **No device found**: Ensure the D435 is plugged and recognized (`rs-enumerate-devices`). On Linux, you may need:
  ```bash
  sudo apt install librealsense2-utils librealsense2-dev
  ```
- **Depth looks wrong**: The script prints the **depth scale** in meters/unit. Recalibrate if necessary.
- **Slow FPS**: Lower `--imgsz` (e.g., 480), reduce classes, or switch to a smaller model (e.g., `yolov8n-seg`).

---

## File Overview
- `realsense_yolov8_seg.py` — Real-time RGB‑D segmentation and distance estimation.
- This README — End-to-end guide: Roboflow training ➜ static-image validation ➜ real-time RGB‑D inference.

**Credits**: Model trained with **Roboflow** dataset and inferred with **Ultralytics YOLOv8**. Camera: **Intel RealSense D435**.
