"""
Stage 1: Image preprocessing for laser dot visibility enhancement.

Applies white balance correction, CLAHE contrast enhancement, and
bilateral denoising to make faint laser dots more detectable.
"""

import numpy as np
import cv2


def gray_world_white_balance(img_bgr):
    """Correct color cast via Gray World assumption (mean of each channel -> gray)."""
    mean_b, mean_g, mean_r = [img_bgr[:, :, c].mean() for c in range(3)]
    gray = (mean_b + mean_g + mean_r) / 3.0
    scale = np.array([gray / mean_b, gray / mean_g, gray / mean_r])
    balanced = img_bgr.astype(np.float32) * scale[np.newaxis, np.newaxis, :]
    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_clahe(img_bgr, clip_limit=3.0, tile_size=8):
    """Boost local contrast on the L channel of LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def bilateral_denoise(img_bgr, d=7, sigma_color=50, sigma_space=50):
    """Edge-preserving denoising that keeps laser dot edges sharp."""
    return cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)


def preprocess(img_bgr, white_balance=True, clahe=True, denoise=True,
               clahe_clip=3.0, clahe_tile=8):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    img_bgr : np.ndarray  (H, W, 3) uint8 BGR image
    white_balance : bool   apply gray-world white balance
    clahe : bool           apply CLAHE on LAB L-channel
    denoise : bool         apply bilateral denoising

    Returns
    -------
    np.ndarray  (H, W, 3) uint8 BGR preprocessed image
    """
    out = img_bgr.copy()
    if white_balance:
        out = gray_world_white_balance(out)
    if clahe:
        out = apply_clahe(out, clip_limit=clahe_clip, tile_size=clahe_tile)
    if denoise:
        out = bilateral_denoise(out)
    return out


def load_image(path):
    """Load image from any supported format (including webp) as BGR uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image as PILImage
        pil = PILImage.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img
