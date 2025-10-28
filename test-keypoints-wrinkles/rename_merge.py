#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe packaging script for synthetic scene dataset.

- DOES NOT modify your original folders (images, corners, masks, depth, depth_normalized).
- Creates a staging folder synthetic_scene_pack/ with:
    * Copied files renamed with mirror effect (first → last, etc.)
    * Numeric filenames only (000001.png, 000002.png, ...), extension preserved.
- Builds a single JSON file corner-line-coordinates.json using corners/masks JSON data.
- Produces a ZIP synthetic-scene.zip with:
    * The staged folders (images, corners, masks, depth, depth_normalized)
    * The final JSON file only (no other .json inside the subfolders).
"""

import os
import json
import shutil
import zipfile
from collections import OrderedDict

# === CONFIG ===
BASE_DIR = os.getcwd()
SRC_FOLDERS = ["images", "corners", "masks", "depth", "depth_normalized"]
PACK_DIR = os.path.join(BASE_DIR, "synthetic_scene_pack")  # staging folder
FINAL_JSON = os.path.join(BASE_DIR, "corner-line-coordinates.json")
ZIP_NAME = os.path.join(BASE_DIR, "synthetic-scene.zip")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")

# Expected source JSONs (originals are never deleted)
CORNERS_JSON_SRC = os.path.join(BASE_DIR, "corners", "keypoints.json")
MASKS_JSON_SRC   = os.path.join(BASE_DIR, "masks", "lines.json")


# ---------- Helpers ----------
def list_media(folder):
    """Return sorted list of media files in a folder (by name)."""
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(IMG_EXTS)
    ])

def ensure_dirs():
    """Create staging folder and its subfolders."""
    os.makedirs(PACK_DIR, exist_ok=True)
    for d in SRC_FOLDERS:
        os.makedirs(os.path.join(PACK_DIR, d), exist_ok=True)

def pad_num(idx, width):
    """Return string with zero padding (e.g., 1 → '000001')."""
    return str(idx).zfill(width)

def mirror_copy_folder(src_folder, ref_width=None):
    """
    Copy files from src_folder into PACK_DIR/src_folder with mirrored numbering.
    Example: first file becomes last number, last file becomes '000001'.
    Returns:
        mapping: { old_name.ext : '000123' } (without extension in value)
        total   : number of files
        ext_map : { old_name.ext : extension }
    """
    src = os.path.join(BASE_DIR, src_folder)
    dst = os.path.join(PACK_DIR, src_folder)

    if not os.path.isdir(src):
        return {}, 0, {}

    files = list_media(src)
    n = len(files)
    if n == 0:
        return {}, 0, {}

    width = ref_width if ref_width is not None else len(str(n))
    mapping = {}
    ext_map = {}

    for i, old_name in enumerate(files):
        # Mirror numbering: first gets last number, second gets second-last, etc.
        new_num = pad_num(n - i, width)
        ext = os.path.splitext(old_name)[1].lower()

        old_path = os.path.join(src, old_name)
        new_path = os.path.join(dst, f"{new_num}{ext}")

        shutil.copy2(old_path, new_path)  # copy file instead of renaming
        mapping[old_name] = new_num
        ext_map[old_name] = ext

    print(f"[INFO] {src_folder}: copied {n} files into {dst}")
    return mapping, n, ext_map

def load_json_safe(path):
    """Safely load JSON if it exists, otherwise return empty dict."""
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load JSON {path}: {e}")
    return {}

def build_final_json(images_mapping, corners_data, masks_data):
    """
    Build final JSON with numeric keys.
    - images_mapping: { old_img_name.ext : '000123' }
    - corners_data:   dict from original corners JSON
    - masks_data:     dict from original masks JSON
    Each entry looks like:
    {
        "000123": {
            "keypoints": [...],
            "line": {"start": [x,y], "end": [x,y]}
        }
    }
    """
    ordered_pairs = sorted(images_mapping.items(), key=lambda kv: int(kv[1]))
    merged = OrderedDict()

    for old_name, num_key in ordered_pairs:
        kps = corners_data.get(old_name)
        line = masks_data.get(old_name)
        entry = {
            "keypoints": kps if isinstance(kps, list) else [],
            "line": line if isinstance(line, dict) else {"start": None, "end": None}
        }
        merged[num_key] = entry

    with open(FINAL_JSON, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"[INFO] Final JSON written: {FINAL_JSON} ({len(merged)} entries)")

def make_zip_without_extra_jsons():
    """
    Create synthetic-scene.zip including:
    - synthetic_scene_pack/<folders> (only media, no JSON inside)
    - corner-line-coordinates.json at root
    """
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in SRC_FOLDERS:
            root = os.path.join(PACK_DIR, folder)
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(".json"):
                        continue  # exclude any JSON
                    fpath = os.path.join(dirpath, fn)
                    arcname = os.path.relpath(fpath, PACK_DIR)
                    zf.write(fpath, arcname)

        if os.path.isfile(FINAL_JSON):
            zf.write(FINAL_JSON, os.path.basename(FINAL_JSON))

    print(f"[INFO] ZIP created: {ZIP_NAME}")


# ---------- Main ----------
def main():
    print("✅ Safe pack start (original folders are untouched).")
    ensure_dirs()

    # Mirror-copy images first (reference numbering)
    images_mapping, n_images, _ = mirror_copy_folder("images")
    if n_images == 0:
        print("[ERROR] No images found in ./images. Aborting.")
        return
    width = len(str(n_images))

    # Mirror-copy other folders with the same width
    for folder in ["corners", "masks", "depth", "depth_normalized"]:
        mirror_copy_folder(folder, ref_width=width)

    # Load original JSONs (they are NOT deleted)
    corners_data = load_json_safe(CORNERS_JSON_SRC)
    masks_data   = load_json_safe(MASKS_JSON_SRC)

    # Build final JSON using numeric keys
    build_final_json(images_mapping, corners_data, masks_data)

    # Create zip archive with staged data + final JSON
    make_zip_without_extra_jsons()

    print("🎉 Done. Check synthetic_scene_pack/ and synthetic-scene.zip")

if __name__ == "__main__":
    main()
