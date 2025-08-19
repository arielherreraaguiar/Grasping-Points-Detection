# CV Grasping Points Detection in Textiles

This project focuses on identifying the best grasping points to unfold a piece of cloth using a robotic arm.  
The chosen strategy is to detect **keypoints at the corners of the cloth** (up to 4 points if all are visible).  

The repository allows you to **generate a custom dataset of towels (folded and flat) in Blender**, process it into COCO format, and then **train a deep learning model based on heatmaps** to detect keypoints.  
A pretrained model is also provided for evaluation and real-time testing with an Intel RealSense RGBD camera.

---

## 📂 Repository Structure

### `Towel-scenes-custom/`
This directory contains all scripts for **dataset generation and preprocessing**.

- **`cloth-blender_custom.py`**  
  Main script to generate synthetic towel images using **Blender 2.80**.  
  - Saves rendered images in `images/`  
  - Generates a **COCO-style JSON file** `dataset_coco.json` inside `images/`, which stores the **corner positions in pixel coordinates**  

- **`draw_keypoints.py`**  
  Visualization tool to validate the JSON annotations.  
  - Creates a directory `corners/`  
  - Saves all images from `images/`, this time with **red dots marking the towel corners**

- **`split_dataset.py`**  
  Splits the dataset into `final_dataset/` with three subfolders:  
  - `train/` → 80% of the images  
  - `val/` → 20% of the images  
  - `test/` → remaining 20% of the images  
  Each folder contains its own **`dataset_coco.json`** file.  

> 💡 Tip: For training, it is recommended to compress the folder `final_dataset/` as `final_dataset.zip` and upload it to Google Drive.  

- **Google Colab Notebook for Training**  
  Train the deep learning model using the generated dataset:  
  [Training Notebook](https://colab.research.google.com/drive/1vDwQTSIpFn1aj6c4Ux8AAtADVSatG2aI?usp=sharing)

---

### Main Directory
Outside `Towel-scenes-custom/`, you will find **model and evaluation scripts**.

- **`best_heatmap.pth`**  
  Pretrained model (trained on the generated dataset).  

- **`eval.py`**  
  Script to evaluate the pretrained model on a dataset.  

- **`realtime_keypoints.py`**  
  Real-time visualization of detected keypoints using an **Intel RealSense RGBD camera**.  

---

## 🖼️ Example Outputs

- **Synthetic Dataset Example (Blender-generated)**  
  ![Example Image](docs/example_dataset.png)

- **Corners Visualization (`draw_keypoints.py`)**  
  ![Corners](docs/example_corners.png)

- **Real-Time Keypoints Detection**  
  ![Realtime](docs/example_realtime.png)

---

## 🚀 How to Run

### 1. Dataset Generation
Run the Blender generator (requires **Blender 2.80**):

```bash
blender2.8 -b -P cloth-blender_custom.py
```

### 2. Keypoints Visualization
```bash
python draw_keypoints.py
```

### 3. Dataset Split
```bash
python split_dataset.py
```

### 4. Model Training (Google Colab)
Follow the notebook:  
[Training Notebook](https://colab.research.google.com/drive/1vDwQTSIpFn1aj6c4Ux8AAtADVSatG2aI?usp=sharing)

### 5. Evaluation
```bash
python eval.py
```

### 6. Real-Time Detection
```bash
python realtime_keypoints.py
```

---

## 🔧 Environment

- Blender **2.80** (Linux recommended)  
- Python 3.8+  
- PyTorch, NumPy, OpenCV  
- Intel RealSense SDK (for real-time keypoints detection)  

---

## 📚 Attribution

This repository adapts and extends the original towel simulation code from:  
[https://github.com/priyasundaresan/cloth-rendering](https://github.com/priyasundaresan/cloth-rendering)

---

## ✨ Summary

- Generate **synthetic datasets** of towels (folded and flat) with Blender  
- Annotate corners in **COCO format**  
- Visualize dataset with **red corner keypoints**  
- Split into `train/`, `val/`, `test/`  
- Train a **heatmap-based deep learning model**  
- Evaluate and test **real-time detection** with a RealSense camera  

---
