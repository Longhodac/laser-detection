"""
Stage 2: Multi-color-space laser scoring.

Produces a unified laser_likelihood map from HSV, LAB, and RGB analysis.
Supports both red and green lasers with automatic color detection.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Hue ranges (OpenCV HSV: H in [0,180], S/V in [0,255])
# ---------------------------------------------------------------------------
RED_HUE_RANGES = [(0, 10), (170, 180)]
GREEN_HUE_RANGES = [(35, 85)]


def _robust_normalize(score, lower_q=95.0, upper_q=99.9):
    """
    Normalize a score image using robust percentiles instead of a raw max.

    This keeps a single saturated region from flattening the dynamic range of
    the rest of the image.
    """
    score_f = score.astype(np.float32)
    lo = float(np.percentile(score_f, lower_q))
    hi = float(np.percentile(score_f, upper_q))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = float(score_f.max())

    if hi <= lo + 1e-6:
        lo = float(score_f.min())
        hi = float(score_f.max())

    if hi <= lo + 1e-6:
        normalized = np.clip(score_f, 0.0, 1.0)
    else:
        normalized = np.clip((score_f - lo) / (hi - lo), 0.0, 1.0)

    stats = {
        "raw_min": float(score_f.min()),
        "raw_max": float(score_f.max()),
        "raw_mean": float(score_f.mean()),
        "raw_std": float(score_f.std()),
        "raw_q95": float(np.percentile(score_f, 95)),
        "raw_q99": float(np.percentile(score_f, 99)),
        "raw_q99_9": float(np.percentile(score_f, 99.9)),
        "robust_lower_q": float(lower_q),
        "robust_upper_q": float(upper_q),
        "robust_lower_value": float(lo),
        "robust_upper_value": float(hi),
    }
    return normalized.astype(np.float32), stats


def _hsv_laser_score(hsv, hue_ranges, sat_min=80, val_min=80):
    """
    Score each pixel by how laser-like it is in HSV space.

    Lasers are highly saturated and bright with specific hue.
    score = S_norm * V_norm * hue_mask
    """
    h, s, v = cv2.split(hsv)
    hue_mask = np.zeros(h.shape, dtype=np.float32)
    for lo, hi in hue_ranges:
        hue_mask[(h >= lo) & (h <= hi)] = 1.0

    s_norm = s.astype(np.float32) / 255.0
    v_norm = v.astype(np.float32) / 255.0

    sat_mask = (s >= sat_min).astype(np.float32)
    val_mask = (v >= val_min).astype(np.float32)

    score = s_norm * v_norm * hue_mask * sat_mask * val_mask
    return score


def _lab_laser_score(lab, color="red"):
    """
    Score pixels using LAB color space deviation.

    Red lasers: strong positive a* (red-green axis)
    Green lasers: strong negative a* or strong negative b* won't work well;
                  instead use deviation from local background in a* channel.
    """
    l, a, b = cv2.split(lab)
    a_f = a.astype(np.float32) - 128.0  # center around 0
    b_f = b.astype(np.float32) - 128.0

    if color == "red":
        score = np.clip(a_f / 127.0, 0, 1)
    elif color == "green":
        score = np.clip(-a_f / 127.0, 0, 1)
    else:
        score_red = np.clip(a_f / 127.0, 0, 1)
        score_green = np.clip(-a_f / 127.0, 0, 1)
        score = np.maximum(score_red, score_green)

    l_norm = l.astype(np.float32) / 255.0
    return score * l_norm


def _rgb_laser_score(bgr_float, color="red"):
    """
    Simple channel-difference scores (generalization of the original red_score).

    Red:   r - 0.5*g - 0.5*b
    Green: g - 0.5*r - 0.5*b
    """
    b, g, r = bgr_float[:, :, 0], bgr_float[:, :, 1], bgr_float[:, :, 2]
    if color == "red":
        score = r - 0.5 * g - 0.5 * b
    elif color == "green":
        score = g - 0.5 * r - 0.5 * b
    else:
        score_red = r - 0.5 * g - 0.5 * b
        score_green = g - 0.5 * r - 0.5 * b
        score = np.maximum(score_red, score_green)
    return np.clip(score, 0, 1)


def _local_contrast_score(gray_float, kernel_size=31):
    """
    Highlight pixels that are locally much brighter than their surroundings.
    Useful for laser dots that are small bright spots on any background.
    """
    local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
    local_std = cv2.blur((gray_float - local_mean) ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(local_std + 1e-8)
    contrast = (gray_float - local_mean) / (local_std + 1e-8)
    return np.clip(contrast / 5.0, 0, 1)


def detect_laser_color(bgr, quantile=99.9):
    """
    Auto-detect whether the dominant laser is red or green.

    Compares robust high-quantiles of red_score and green_score so a single
    outlier pixel does not dominate the decision.
    """
    bgr_f = bgr.astype(np.float32) / 255.0
    b, g, r = bgr_f[:, :, 0], bgr_f[:, :, 1], bgr_f[:, :, 2]

    red_score = np.clip(r - 0.5 * g - 0.5 * b, 0, 1)
    green_score = np.clip(g - 0.5 * r - 0.5 * b, 0, 1)

    red_q = float(np.percentile(red_score, quantile))
    green_q = float(np.percentile(green_score, quantile))

    if red_q >= green_q:
        return "red"
    return "green"


def compute_laser_score(bgr, color="auto",
                        w_hsv=0.35, w_lab=0.25, w_rgb=0.25, w_contrast=0.15,
                        robust_lower_q=95.0, robust_upper_q=99.9,
                        return_debug=False):
    """
    Compute a unified laser likelihood map by fusing HSV, LAB, RGB, and
    local contrast scores.

    Parameters
    ----------
    bgr : np.ndarray  (H, W, 3) uint8 BGR image (preprocessed)
    color : str        "red", "green", or "auto"
    w_hsv, w_lab, w_rgb, w_contrast : float  fusion weights

    Returns
    -------
    score_map : np.ndarray (H, W) float32 in [0, 1]
    detected_color : str   "red" or "green"
    """
    if color == "auto":
        color = detect_laser_color(bgr, quantile=robust_upper_q)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    bgr_f = bgr.astype(np.float32) / 255.0
    gray_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    hue_ranges = RED_HUE_RANGES if color == "red" else GREEN_HUE_RANGES

    s_hsv = _hsv_laser_score(hsv, hue_ranges)
    s_lab = _lab_laser_score(lab, color)
    s_rgb = _rgb_laser_score(bgr_f, color)
    s_contrast = _local_contrast_score(gray_f)

    fused_raw = (w_hsv * s_hsv + w_lab * s_lab + w_rgb * s_rgb +
                 w_contrast * s_contrast)
    fused, score_stats = _robust_normalize(
        fused_raw,
        lower_q=robust_lower_q,
        upper_q=robust_upper_q,
    )

    if return_debug:
        debug_info = {
            "raw_score_map": fused_raw.astype(np.float32),
            "score_stats": score_stats,
            "score_weights": {
                "hsv": float(w_hsv),
                "lab": float(w_lab),
                "rgb": float(w_rgb),
                "contrast": float(w_contrast),
            },
            "color_quantile": float(robust_upper_q),
        }
        return fused.astype(np.float32), color, debug_info

    return fused.astype(np.float32), color
