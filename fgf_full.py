"""
Stage 4: Full Fast Gaussian Fitting (FGF) from Wan et al. 2018
"Star Centroiding Based on Fast Gaussian Fitting for Star Sensors"

Implements the complete two-pass algorithm from Section 2.4:
  Pass 1: rough FGF on top-N pixels -> estimate Gaussian params -> per-pixel SNR
  Pass 2: reselect pixels with SNR > T -> accurate FGF

The closed-form solver uses the log-intensity linear least-squares formulation
from Equations 12-17 of the paper.
"""

import numpy as np


def _build_coordinates(h, w):
    """Return (x_grid, y_grid) meshgrids for a (h, w) patch."""
    return np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))


def _solve_fgf_linear(pixels_x, pixels_y, pixels_I):
    """
    Closed-form Fast Gaussian Fitting via linear least squares.

    Solves Equations 14-17 from the paper:
        min_{m,n,p,q,k} sum_i [m*I_i*x_i^2 + n*I_i*y_i^2
                                + p*I_i*x_i + q*I_i*y_i
                                + k*I_i + I_i*ln(I_i)]^2

    where m = 1/(2*sx^2), n = 1/(2*sy^2), p = -xc/sx^2, q = -yc/sy^2,
          k = xc^2/(2*sx^2) + yc^2/(2*sy^2) - ln(A)

    Returns (xc, yc, sigma_x, sigma_y, A) or None if solver fails.
    """
    I = pixels_I.copy()
    I = np.maximum(I, 1e-10)  # avoid log(0)

    x = pixels_x
    y = pixels_y

    Ix2 = I * x * x
    Iy2 = I * y * y
    Ix = I * x
    Iy = I * y
    IlnI = I * np.log(I)

    # Build the 5-column design matrix A and target vector b
    # h_i = m*I*x^2 + n*I*y^2 + p*I*x + q*I*y + k*I + I*ln(I) = 0
    # Rearrange: [I*x^2, I*y^2, I*x, I*y, I] * [m,n,p,q,k]^T = -I*ln(I)
    A_mat = np.column_stack([Ix2, Iy2, Ix, Iy, I])
    b_vec = -IlnI

    try:
        result, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None

    m, n, p, q, k = result

    if m <= 0 or n <= 0:
        return None

    # Recover Gaussian parameters (Eq 17)
    xc = -p / (2 * m)
    yc = -q / (2 * n)
    sigma_x = 1.0 / np.sqrt(2 * m)
    sigma_y = 1.0 / np.sqrt(2 * n)
    A = np.exp(p ** 2 / (4 * m) + q ** 2 / (4 * n) - k)

    if sigma_x <= 0 or sigma_y <= 0 or A <= 0:
        return None
    if np.isnan(xc) or np.isnan(yc):
        return None

    return xc, yc, sigma_x, sigma_y, A


def _weighted_centroid(pixels_x, pixels_y, pixels_I):
    """Fallback: simple intensity-weighted centroid (like the original fgf)."""
    W = pixels_I.sum()
    if W < 1e-10:
        return np.mean(pixels_x), np.mean(pixels_y), 1.0, 1.0, 0.0

    xc = np.dot(pixels_I, pixels_x) / W
    yc = np.dot(pixels_I, pixels_y) / W
    sx = np.sqrt(np.dot(pixels_I, (pixels_x - xc) ** 2) / W)
    sy = np.sqrt(np.dot(pixels_I, (pixels_y - yc) ** 2) / W)
    sx = max(sx, 0.5)
    sy = max(sy, 0.5)
    A = pixels_I.max()
    return xc, yc, sx, sy, A


def fgf_full(image, n_init=5, snr_threshold=3.0):
    """
    Full two-pass FGF algorithm from Section 2.4 of the paper.

    Parameters
    ----------
    image : np.ndarray (H, W)  grayscale patch, float, values in [0, 1]
    n_init : int                number of brightest pixels for initial pass
    snr_threshold : float       SNR threshold T for pixel reselection (paper: T=3)

    Returns
    -------
    xc, yc : float       sub-pixel centroid coordinates in patch
    sigma_x, sigma_y : float   Gaussian widths
    A : float             amplitude
    confidence : float    quality metric in [0, 1]
    """
    h, w = image.shape
    x_grid, y_grid = _build_coordinates(h, w)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    I_flat = image.flatten().astype(np.float64)

    # ---------------------------------------------------------------
    # Pass 1: Initial selection of pixels (top-N by intensity)
    # ---------------------------------------------------------------
    top_idx = np.argsort(I_flat)[-n_init:]
    init_x = x_flat[top_idx]
    init_y = y_flat[top_idx]
    init_I = I_flat[top_idx]

    result1 = _solve_fgf_linear(init_x, init_y, init_I)
    if result1 is None:
        result1 = _weighted_centroid(init_x, init_y, init_I)

    xc1, yc1, sx1, sy1, A1 = result1

    # ---------------------------------------------------------------
    # Estimate per-pixel noise and SNR (Equations 18-20)
    # ---------------------------------------------------------------
    S_hat = A1 * np.exp(
        -((x_flat - xc1) ** 2) / (2 * sx1 ** 2)
        - ((y_flat - yc1) ** 2) / (2 * sy1 ** 2)
    )
    N_hat = I_flat - S_hat  # Eq 19
    snr = np.abs(S_hat / (np.abs(N_hat) + 1e-10))  # Eq 20

    # ---------------------------------------------------------------
    # Pass 2: Reselect pixels with SNR > T (Eq 21)
    # ---------------------------------------------------------------
    snr_mask = snr > snr_threshold
    # Also require positive intensity
    valid_mask = snr_mask & (I_flat > 1e-10)

    if valid_mask.sum() < 5:
        # Fall back: use top-N brightest in a 3-sigma window around pass-1 center
        dist_mask = (
            (np.abs(x_flat - xc1) < 3 * sx1) &
            (np.abs(y_flat - yc1) < 3 * sy1) &
            (I_flat > 1e-10)
        )
        if dist_mask.sum() >= 5:
            valid_mask = dist_mask
        else:
            valid_mask = I_flat > np.percentile(I_flat, 80)

    sel_x = x_flat[valid_mask]
    sel_y = y_flat[valid_mask]
    sel_I = I_flat[valid_mask]

    result2 = _solve_fgf_linear(sel_x, sel_y, sel_I)
    if result2 is None:
        result2 = _weighted_centroid(sel_x, sel_y, sel_I)

    xc2, yc2, sx2, sy2, A2 = result2

    # ---------------------------------------------------------------
    # Confidence scoring
    # ---------------------------------------------------------------
    confidence = _compute_confidence(image, xc2, yc2, sx2, sy2, A2)

    return xc2, yc2, sx2, sy2, A2, confidence


def _compute_confidence(image, xc, yc, sx, sy, A):
    """
    Compute a quality/confidence metric for the fit.

    Based on:
    - Goodness of fit (R-squared)
    - Peak SNR of the fitted spot vs background
    - Compactness (sigma reasonableness and aspect ratio)
    - Center location within the patch
    """
    h, w = image.shape
    x_grid, y_grid = _build_coordinates(h, w)

    fitted = A * np.exp(
        -((x_grid - xc) ** 2) / (2 * sx ** 2)
        - ((y_grid - yc) ** 2) / (2 * sy ** 2)
    )
    residual = image - fitted
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((image - image.mean()) ** 2) + 1e-10
    r_squared = max(0, 1.0 - ss_res / ss_tot)

    bg_mask = fitted < 0.1 * A
    if bg_mask.sum() > 0:
        bg_std = image[bg_mask].std() + 1e-10
        peak_snr = A / bg_std
    else:
        peak_snr = A / (image.std() + 1e-10)
    snr_score = min(1.0, peak_snr / 20.0)

    min_dim = min(h, w)
    sigma_score = 1.0
    # Laser dots should be compact relative to patch size
    if sx > min_dim / 4 or sy > min_dim / 4:
        sigma_score *= 0.3
    elif sx > min_dim / 6 or sy > min_dim / 6:
        sigma_score *= 0.6
    if sx < 0.3 or sy < 0.3:
        sigma_score *= 0.5
    # Penalize extreme aspect ratios (laser dots are roughly circular)
    aspect = max(sx, sy) / (min(sx, sy) + 1e-10)
    if aspect > 5:
        sigma_score *= 0.3
    elif aspect > 3:
        sigma_score *= 0.6

    center_score = 1.0
    if xc < 0 or xc >= w or yc < 0 or yc >= h:
        center_score = 0.1
    else:
        # Slightly penalize detections far from patch center
        dist_from_center = np.sqrt((xc - w / 2) ** 2 + (yc - h / 2) ** 2)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        center_score = max(0.3, 1.0 - 0.5 * dist_from_center / max_dist)

    confidence = (0.35 * r_squared + 0.30 * snr_score +
                  0.20 * sigma_score + 0.15 * center_score)
    return float(np.clip(confidence, 0, 1))


def fgf_simple(image, n_pass1=5):
    """
    Original simplified FGF (preserved for comparison).
    Two-pass weighted centroid without the full linear LS solver.
    """
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    flat = image.flatten()
    top_idx = np.argsort(flat)[-n_pass1:]
    wx = x.flatten()[top_idx]
    wy = y.flatten()[top_idx]
    wi = flat[top_idx]

    W = wi.sum()
    xc1 = np.dot(wi, wx) / W
    yc1 = np.dot(wi, wy) / W
    sigx1 = np.sqrt(np.dot(wi, (wx - xc1) ** 2) / W)
    sigy1 = np.sqrt(np.dot(wi, (wy - yc1) ** 2) / W)
    sigx1 = max(sigx1, 0.5)
    sigy1 = max(sigy1, 0.5)

    mask = (np.abs(x - xc1) < 3 * sigx1) & (np.abs(y - yc1) < 3 * sigy1)
    wx2 = x[mask]
    wy2 = y[mask]
    wi2 = image[mask]

    W2 = wi2.sum()
    xc2 = np.dot(wi2, wx2) / W2
    yc2 = np.dot(wi2, wy2) / W2
    sigx2 = np.sqrt(np.dot(wi2, (wx2 - xc2) ** 2) / W2)
    sigy2 = np.sqrt(np.dot(wi2, (wy2 - yc2) ** 2) / W2)
    A_est = np.max(image)

    return xc2, yc2, sigx2, sigy2, A_est
