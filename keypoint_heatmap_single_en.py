# keypoint_heatmap_single_en.py
# ------------------------------------------------------------
# HeatmapNet: pequeño U-Net que produce 1 heatmap (esquinas visibles como picos).
# Coincide con la arquitectura de tu notebook (encoders 64/128/256, stride=4).
# Incluye utilidades: topk_peaks y order_tl_tr_br_bl.
# ------------------------------------------------------------

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tamaño de entrada y stride como en tu notebook
IMG_SIZE = 256
HEAT_STRIDE = 4  # -> heatmap de 64x64

# -------------------------
# Bloques (tipo U-Net)
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
    """U-Net pequeño → 1 heatmap (logits) reescalado a (IMG_SIZE/HEAT_STRIDE)."""
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

        # Decoder (ojo con canales de concat)
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
        # Reescala a 64x64 (stride 4) como en tu notebook
        y  = F.interpolate(
            y,
            size=(IMG_SIZE // HEAT_STRIDE, IMG_SIZE // HEAT_STRIDE),
            mode='bilinear', align_corners=False
        )
        return y  # logits (no sigmoid)

# -------------------------
# Utilidades
# -------------------------
def topk_peaks(heat_np: np.ndarray, K: int = 4, thresh: float = 0.25):
    """
    Extrae hasta K picos [x,y,score] de un heatmap (numpy) en [0,1].
    heat_np: (H,W) después de sigmoid.
    """
    H, W = heat_np.shape
    flat = heat_np.reshape(-1)
    idxs = np.argsort(-flat)  # descendente
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
    Heurística para ordenar 4 puntos como (TL, TR, BR, BL).
    Devuelve array Nx2; si N!=4, retorna entrada sin cambios.
    """
    if points_xy is None or len(points_xy) != 4:
        return points_xy
    pts = np.array(points_xy, dtype=np.float32)
    # Ordena por Y (arriba->abajo), luego por X
    idx = np.lexsort((pts[:,0], pts[:,1]))
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2,0])]
    bot = pts[2:][np.argsort(pts[2:,0])]
    TL, TR = top[0], top[1]
    BL, BR = bot[0], bot[1]
    return np.stack([TL, TR, BR, BL], axis=0)

# Autotest rápido
if __name__ == "__main__":
    net = HeatmapNet(out_ch=1)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", tuple(y.shape))  # (1,1,64,64)
