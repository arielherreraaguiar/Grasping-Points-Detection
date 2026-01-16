# keypoint_heatmap_single_en.py
# ------------------------------------------------------------
# HeatmapNet: small U-Net that produces 1 heatmap (visible corners as peaks).
# Matches the architecture of your notebook (encoders 64/128/256, stride=4).
# Includes utilities: topk_peaks and order_tl_tr_br_bl.
# ------------------------------------------------------------

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input size and stride as in your notebook
IMG_SIZE = 256
HEAT_STRIDE = 4  # -> heatmap of 64x64

# -------------------------
# Blocks (U-Net type)
# -------------------------
def _block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class HeatmapNet(nn.Module):
    """Small U-Net → 1 heatmap (logits) rescaled to (IMG_SIZE/HEAT_STRIDE)."""
    def __init__(self, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = _block(3, 64)        # 256 -> 256
        self.pool1 = nn.MaxPool2d(2)     # 256 -> 128
        self.enc2 = _block(64, 128)
        self.pool2 = nn.MaxPool2d(2)     # 128 -> 64
        self.enc3 = _block(128, 256)
        self.pool3 = nn.MaxPool2d(2)     # 64  -> 32
        self.bott = _block(256, 256)

        # Decoder (be careful with concat channels)
        self.up1  = nn.ConvTranspose2d(256, 128, 2, stride=2)      # 32 -> 64
        self.dec1 = _block(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)       # 64 -> 128
        self.dec2 = _block(64 + 128, 64)

        self.up3  = nn.ConvTranspose2d(64, 64, 2, stride=2)        # 128 -> 256
        self.dec3 = _block(64 + 64, 64)

        self.out = nn.Conv2d(64, out_ch, 1)  # logits

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b  = self.bott(p3)

        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e3], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, e1], dim=1))

        y  = self.out(d3)  # [B,1,256,256] logits
        # Rescale to 64x64 (stride 4) as in your notebook
        y  = F.interpolate(
            y,
            size=(IMG_SIZE // HEAT_STRIDE, IMG_SIZE // HEAT_STRIDE),
            mode='bilinear', align_corners=False
        )
        return y  # logits (no sigmoid)

# -------------------------
# Utilities
# -------------------------
def topk_peaks(heat_np: np.ndarray, K: int = 4, thresh: float = 0.25):
    """
    Extract up to K peaks [x,y,score] from a heatmap (numpy) in [0,1].
    heat_np: (H,W) after sigmoid.
    """
    H, W = heat_np.shape
    flat = heat_np.reshape(-1)
    idxs = np.argsort(-flat)  # descending
    peaks = []
    taken = np.zeros_like(flat, dtype=bool)
    for idx in idxs:
        if flat[idx] < thresh:
            break
        if taken[idx]:
            continue
        y, x = divmod(idx, W)
        peaks.append([x, y, float(flat[idx])])
        if len(peaks) == K:
            break
        taken[idx] = True
    return np.array(peaks, dtype=np.float32)

def order_tl_tr_br_bl(points_xy: np.ndarray) -> np.ndarray:
    """
    Heuristic to order 4 points as (TL, TR, BR, BL).
    Returns Nx2 array; if N!=4, returns input unchanged.
    """
    if points_xy is None or len(points_xy) != 4:
        return points_xy
    pts = np.array(points_xy, dtype=np.float32)
    # Sort by Y (top->bottom), then by X
    idx = np.lexsort((pts[:,0], pts[:,1]))
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2,0])]
    bot = pts[2:][np.argsort(pts[2:,0])]
    TL, TR = top[0], top[1]
    BL, BR = bot[0], bot[1]
    return np.stack([TL, TR, BR, BL], axis=0)

# Quick autotest
if __name__ == "__main__":
    net = HeatmapNet(out_ch=1)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", tuple(y.shape))  # (1,1,64,64)

