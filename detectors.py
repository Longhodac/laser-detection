"""
Stage 3: Candidate detection using multiple complementary algorithms.

Combines LoG/DoG blob detection, morphological top-hat, and fast radial
symmetry transform, with NMS fusion to produce ranked candidates.
"""

import numpy as np
import cv2
from skimage.feature import blob_log, blob_dog


# ---------------------------------------------------------------------------
# LoG / DoG blob detection
# ---------------------------------------------------------------------------

def detect_blobs_log(score_map, min_sigma=1, max_sigma=8, num_sigma=6,
                     threshold=0.1):
    """
    Multi-scale LoG blob detection on the laser score map.

    Returns list of (y, x, sigma, response) tuples.
    """
    blobs = blob_log(score_map, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold)
    results = []
    for blob in blobs:
        y, x, sigma = blob
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < score_map.shape[0] and 0 <= xi < score_map.shape[1]:
            response = score_map[yi, xi]
            results.append((y, x, sigma, float(response)))
    return results


def detect_blobs_dog(score_map, min_sigma=1, max_sigma=8, sigma_ratio=1.6,
                     threshold=0.1):
    """
    Multi-scale DoG blob detection (faster approximation of LoG).
    """
    blobs = blob_dog(score_map, min_sigma=min_sigma, max_sigma=max_sigma,
                     sigma_ratio=sigma_ratio, threshold=threshold)
    results = []
    for blob in blobs:
        y, x, sigma = blob
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < score_map.shape[0] and 0 <= xi < score_map.shape[1]:
            response = score_map[yi, xi]
            results.append((y, x, sigma, float(response)))
    return results


# ---------------------------------------------------------------------------
# Morphological white top-hat
# ---------------------------------------------------------------------------

def detect_tophat(score_map, kernel_radius=5, threshold_factor=3.0):
    """
    White top-hat transform to isolate small bright features.

    Uses a circular structuring element; peaks in the top-hat response
    are candidate laser dots.
    """
    score_u8 = (np.clip(score_map, 0, 1) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1)
    )
    tophat = cv2.morphologyEx(score_u8, cv2.MORPH_TOPHAT, kernel)
    tophat_f = tophat.astype(np.float32) / 255.0

    mean_val = tophat_f.mean()
    std_val = tophat_f.std()
    thresh = mean_val + threshold_factor * std_val

    candidates = []
    binary = (tophat_f > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        area = M["m00"]
        sigma_est = np.sqrt(area / np.pi)
        yi, xi = int(round(cy)), int(round(cx))
        if 0 <= yi < score_map.shape[0] and 0 <= xi < score_map.shape[1]:
            response = score_map[yi, xi]
            candidates.append((cy, cx, sigma_est, float(response)))
    return candidates, tophat_f


# ---------------------------------------------------------------------------
# Fast Radial Symmetry Transform  (Loy & Zelinsky, 2003)
# ---------------------------------------------------------------------------

def fast_radial_symmetry(gray, radii, alpha=2.0, std_factor=0.25):
    """
    Detect radially symmetric bright spots at given radii.
    Vectorized implementation of Loy & Zelinsky (2003).

    Parameters
    ----------
    gray : np.ndarray   (H, W) float32 in [0, 1]
    radii : list[int]   radii to check (e.g. [2, 3, 4, 5])
    alpha : float        strictness of radial symmetry (higher = stricter)
    std_factor : float   Gaussian smoothing factor per radius

    Returns
    -------
    S : np.ndarray (H, W) float32  symmetry response map
    """
    h, w = gray.shape
    S_acc = np.zeros((h, w), dtype=np.float64)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-10

    # Precompute coordinate grids
    yy, xx = np.mgrid[:h, :w]

    # Only process pixels with significant gradient
    sig_mask = mag > 1e-4
    src_y = yy[sig_mask]
    src_x = xx[sig_mask]
    src_gx = gx[sig_mask]
    src_gy = gy[sig_mask]
    src_mag = mag[sig_mask]
    norm_gx = src_gx / src_mag
    norm_gy = src_gy / src_mag

    for n in radii:
        O_n = np.zeros((h, w), dtype=np.float64)
        M_n = np.zeros((h, w), dtype=np.float64)

        ppx = np.round(src_x + n * norm_gx).astype(np.intp)
        ppy = np.round(src_y + n * norm_gy).astype(np.intp)

        valid = (ppy >= 0) & (ppy < h) & (ppx >= 0) & (ppx < w)
        vpy = ppy[valid]
        vpx = ppx[valid]
        vmag = src_mag[valid]

        np.add.at(O_n, (vpy, vpx), 1)
        np.add.at(M_n, (vpy, vpx), vmag)

        O_n = np.minimum(O_n, alpha)
        F_n = (M_n / (np.maximum(O_n, 1))) * ((O_n / alpha) ** alpha)

        ksize = max(3, int(2 * np.ceil(n * std_factor) + 1))
        if ksize % 2 == 0:
            ksize += 1
        F_n = cv2.GaussianBlur(F_n, (ksize, ksize), n * std_factor)
        S_acc += F_n

    S_acc = S_acc / len(radii)
    if S_acc.max() > 0:
        S_acc = S_acc / S_acc.max()
    return S_acc.astype(np.float32)


def detect_radial_symmetry(score_map, gray, radii=None, threshold_factor=3.0):
    """
    Detect radially symmetric peaks in the image, filtered by score_map.

    Returns candidates and the raw symmetry map.
    """
    if radii is None:
        radii = [2, 3, 4, 5]

    sym_map = fast_radial_symmetry(gray, radii)
    combined = sym_map * score_map
    if combined.max() > 0:
        combined = combined / combined.max()

    mean_val = combined.mean()
    std_val = combined.std()
    thresh = mean_val + threshold_factor * std_val

    candidates = []
    binary = (combined > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        sigma_est = np.sqrt(M["m00"] / np.pi)
        yi, xi = int(round(cy)), int(round(cx))
        if 0 <= yi < score_map.shape[0] and 0 <= xi < score_map.shape[1]:
            response = combined[yi, xi]
            candidates.append((cy, cx, sigma_est, float(response)))
    return candidates, combined


# ---------------------------------------------------------------------------
# Non-Maximum Suppression & Fusion
# ---------------------------------------------------------------------------

def nms_candidates(candidates, merge_radius=10, max_candidates=20):
    """
    Merge nearby candidates from different detectors via NMS.

    Each candidate is (y, x, sigma, response). Keeps the highest-response
    candidate within each merge_radius neighborhood.
    """
    if not candidates:
        return []

    cands = sorted(candidates, key=lambda c: c[3], reverse=True)
    kept = []
    used = np.zeros(len(cands), dtype=bool)

    for i, (y1, x1, s1, r1) in enumerate(cands):
        if used[i]:
            continue
        used[i] = True
        group_y, group_x, group_s, group_r = [y1], [x1], [s1], [r1]

        for j in range(i + 1, len(cands)):
            if used[j]:
                continue
            y2, x2, s2, r2 = cands[j]
            dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            if dist < merge_radius:
                used[j] = True
                group_y.append(y2)
                group_x.append(x2)
                group_s.append(s2)
                group_r.append(r2)

        weights = np.array(group_r)
        wsum = weights.sum()
        avg_y = np.dot(weights, group_y) / wsum
        avg_x = np.dot(weights, group_x) / wsum
        avg_s = np.dot(weights, group_s) / wsum
        max_r = max(group_r)
        kept.append((avg_y, avg_x, avg_s, max_r))

        if len(kept) >= max_candidates:
            break

    return kept


def run_detectors(score_map, gray, use_log=True, use_tophat=True,
                  use_radial=True, merge_radius=10, max_candidates=20):
    """
    Run all enabled detectors and fuse results via NMS.

    Parameters
    ----------
    score_map : np.ndarray (H, W)  laser likelihood map
    gray : np.ndarray (H, W)       grayscale float32 in [0,1]
    use_log : bool    run LoG blob detection
    use_tophat : bool run morphological top-hat
    use_radial : bool run fast radial symmetry

    Returns
    -------
    candidates : list of (y, x, sigma, response)
    debug_info : dict with intermediate maps
    """
    all_candidates = []
    debug = {}

    if use_log:
        log_cands = detect_blobs_log(score_map)
        dog_cands = detect_blobs_dog(score_map)
        all_candidates.extend(log_cands)
        all_candidates.extend(dog_cands)
        debug["log_count"] = len(log_cands)
        debug["dog_count"] = len(dog_cands)

    if use_tophat:
        tophat_cands, tophat_map = detect_tophat(score_map)
        all_candidates.extend(tophat_cands)
        debug["tophat_map"] = tophat_map
        debug["tophat_count"] = len(tophat_cands)

    if use_radial:
        radial_cands, sym_map = detect_radial_symmetry(score_map, gray)
        all_candidates.extend(radial_cands)
        debug["sym_map"] = sym_map
        debug["radial_count"] = len(radial_cands)

    fused = nms_candidates(all_candidates, merge_radius=merge_radius,
                           max_candidates=max_candidates)
    debug["raw_count"] = len(all_candidates)
    debug["fused_count"] = len(fused)

    return fused, debug
