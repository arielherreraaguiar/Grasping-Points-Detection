# Towel Scene Generator with Blender 2.8

This script generates synthetic images of a deformable towel using Blender 2.8, simulating realistic cloth folding and indoor lighting conditions.

---

## 💡 Features

- Procedural cloth generation with physics simulation
- Random initial positions, rotations, and pinned vertices
- Table surface for realistic interaction
- High-quality fabric textures from the `textures/` folder, Blender, and external sources: **AmbientCG, CGBookcase, PolyHaven**
- Smart coloring logic:
  - First time a texture is used → no tint (original color)
  - If a texture is reused → apply a random color tint
- Indoor lighting using a randomized `POINT` light source

---

## 🖼️ Output

- All images are saved to the `images/` folder
- Each simulation (episode) renders 10 frames (sampled every 3 frames over a 30-frame simulation)
- The full run generates **5000 images** across 500 randomized cloth drops

---

## 🚀 How to Run

Make sure you are using **Blender 2.80** and that the alias `blender2.8` is set in your terminal.

### 1. 📥 Download Blender 2.80 (Linux)

Download from the official archive:  
[https://download.blender.org/release/Blender2.80/](https://download.blender.org/release/Blender2.80/)

Example:

```bash
cd ~/programs
wget https://download.blender.org/release/Blender2.80/blender-2.80-linux-glibc217-x86_64.tar.bz2
tar -xjf blender-2.80-linux-glibc217-x86_64.tar.bz2
mv blender-2.80-linux-glibc217-x86_64 blender2.8
```

### 2. 🧠 Set the alias

Open your `.bashrc` or `.zshrc` and add:

```bash
alias blender2.8="$HOME/programs/blender2.8/blender"
```

Then reload your shell:

```bash
source ~/.bashrc
```

### 3. ✅ Run the generator

```bash
blender2.8 -b -P cloth-blender_custom.py
```

---
## 📚 Attribution

The towel scene generation code is adapted from:  
[https://github.com/priyasundaresan/cloth-rendering](https://github.com/priyasundaresan/cloth-rendering)  
All dataset generation, keypoint handling, splitting, and deep learning model training were implemented in this repository.

---