"""
Laser Detection Pipeline -- orchestrator.

Ties together preprocessing, color scoring, candidate detection,
and full FGF sub-pixel refinement into a single detect_laser() call.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from preprocess import preprocess, load_image
from color_scoring import compute_laser_score
from detectors import run_detectors
from fgf_full import fgf_full


@dataclass
class LaserDetection:
    """Result of a single detected laser dot."""
    x: float               # sub-pixel x in full image
    y: float               # sub-pixel y in full image
    sigma_x: float         # Gaussian width x
    sigma_y: float         # Gaussian width y
    amplitude: float       # fitted peak amplitude
    confidence: float      # quality score [0, 1]
    patch_x0: int = 0      # patch origin x in full image
    patch_y0: int = 0      # patch origin y in full image


@dataclass
class PipelineResult:
    """Full result of the detection pipeline."""
    detections: List[LaserDetection] = field(default_factory=list)
    laser_color: str = "unknown"
    score_map: Optional[np.ndarray] = None
    preprocessed: Optional[np.ndarray] = None
    debug: dict = field(default_factory=dict)


def extract_patch(score_map, cx, cy, pad=30):
    """Extract a square patch centered at (cx, cy) from the score map."""
    h, w = score_map.shape
    cy_i, cx_i = int(round(cy)), int(round(cx))
    y0 = max(cy_i - pad, 0)
    y1 = min(cy_i + pad + 1, h)
    x0 = max(cx_i - pad, 0)
    x1 = min(cx_i + pad + 1, w)
    patch = score_map[y0:y1, x0:x1]
    return patch, x0, y0


def detect_laser(image_path=None, image_bgr=None, laser_color="auto",
                 do_preprocess=True, pad=30, min_confidence=0.15,
                 max_detections=10, debug=False):
    """
    Main pipeline entry point.

    Parameters
    ----------
    image_path : str           path to image file
    image_bgr : np.ndarray     alternatively, pass BGR image directly
    laser_color : str          "red", "green", or "auto"
    do_preprocess : bool       run preprocessing stage
    pad : int                  half-size of patch for FGF refinement
    min_confidence : float     discard detections below this confidence
    max_detections : int       max number of laser dots to return
    debug : bool               include debug maps in result

    Returns
    -------
    PipelineResult
    """
    # --- Load ---
    if image_bgr is not None:
        bgr = image_bgr.copy()
    elif image_path is not None:
        bgr = load_image(image_path)
    else:
        raise ValueError("Provide image_path or image_bgr")

    result = PipelineResult()

    # --- Stage 1: Preprocess ---
    if do_preprocess:
        bgr_proc = preprocess(bgr)
    else:
        bgr_proc = bgr.copy()
    result.preprocessed = bgr_proc

    # --- Stage 2: Color scoring ---
    score_map, detected_color = compute_laser_score(bgr_proc, color=laser_color)
    result.laser_color = detected_color
    result.score_map = score_map

    # --- Stage 3: Candidate detection ---
    gray = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Tune detector sensitivity based on score map statistics
    score_max = score_map.max()
    adaptive_threshold = max(0.02, score_max * 0.1)

    candidates, det_debug = run_detectors(
        score_map, gray,
        use_log=True, use_tophat=True, use_radial=True,
        merge_radius=max(10, pad // 2),
        max_candidates=max_detections * 3
    )

    if debug:
        result.debug = det_debug

    # Always include score-map argmax as a candidate
    cy_peak, cx_peak = np.unravel_index(np.argmax(score_map), score_map.shape)
    peak_val = score_map[cy_peak, cx_peak]
    argmax_cand = (float(cy_peak), float(cx_peak), 2.0, float(peak_val))

    if not candidates:
        candidates = [argmax_cand]
    else:
        # Add argmax if not already near an existing candidate
        already_near = any(
            np.sqrt((c[0] - argmax_cand[0])**2 + (c[1] - argmax_cand[1])**2) < 15
            for c in candidates
        )
        if not already_near:
            candidates.insert(0, argmax_cand)

    # --- Stage 4: FGF refinement on each candidate ---
    # Run FGF on both score_map and raw grayscale, keep the better fit
    detections = []
    for (cy, cx, sigma_est, response) in candidates:
        patch_score, x0, y0 = extract_patch(score_map, cx, cy, pad=pad)
        patch_gray, _, _ = extract_patch(gray, cx, cy, pad=pad)
        if patch_score.shape[0] < 5 or patch_score.shape[1] < 5:
            continue

        try:
            xc_s, yc_s, sx_s, sy_s, A_s, conf_s = fgf_full(
                patch_score.astype(np.float64))
        except Exception:
            conf_s = 0

        try:
            xc_g, yc_g, sx_g, sy_g, A_g, conf_g = fgf_full(
                patch_gray.astype(np.float64))
        except Exception:
            conf_g = 0

        if conf_s >= conf_g and conf_s > 0:
            xc, yc, sx, sy, A, conf = xc_s, yc_s, sx_s, sy_s, A_s, conf_s
        elif conf_g > 0:
            xc, yc, sx, sy, A, conf = xc_g, yc_g, sx_g, sy_g, A_g, conf_g
        else:
            continue

        # Blend FGF confidence with the score-map response at the candidate
        # This ensures candidates at high-scoring locations are ranked higher
        final_conf = 0.6 * conf + 0.4 * response

        det = LaserDetection(
            x=x0 + xc,
            y=y0 + yc,
            sigma_x=sx,
            sigma_y=sy,
            amplitude=A,
            confidence=final_conf,
            patch_x0=x0,
            patch_y0=y0,
        )
        detections.append(det)

    # --- Filter, deduplicate, and sort ---
    detections = [d for d in detections if d.confidence >= min_confidence]
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Post-FGF deduplication: remove detections within merge_dist of a
    # higher-confidence detection
    merge_dist = max(10, pad // 2)
    deduped = []
    for d in detections:
        is_dup = any(
            np.sqrt((d.x - k.x)**2 + (d.y - k.y)**2) < merge_dist
            for k in deduped
        )
        if not is_dup:
            deduped.append(d)

    result.detections = deduped[:max_detections]
    return result


def visualize_result(image_path_or_bgr, result, figsize=(18, 5)):
    """
    Visualize pipeline result with matplotlib.

    Shows: original image with detections, score map, and top detection patch.
    """
    import matplotlib.pyplot as plt

    if isinstance(image_path_or_bgr, str):
        bgr = load_image(image_path_or_bgr)
    else:
        bgr = image_path_or_bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    n_plots = 3 if result.detections else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Original image with detections
    axes[0].imshow(rgb)
    colors = ['yellow', 'cyan', 'magenta', 'lime', 'orange']
    for i, det in enumerate(result.detections):
        c = colors[i % len(colors)]
        axes[0].plot(det.x, det.y, 'o', ms=14, mew=2, fillstyle='none',
                     color=c, label=f'#{i+1} conf={det.confidence:.2f}')
        theta = np.linspace(0, 2 * np.pi, 100)
        axes[0].plot(det.x + 2 * det.sigma_x * np.cos(theta),
                     det.y + 2 * det.sigma_y * np.sin(theta),
                     '--', color=c, lw=1.5)
    axes[0].set_title(f'Detections ({result.laser_color} laser)\n'
                      f'{len(result.detections)} found')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].axis('off')

    # Score map
    im1 = axes[1].imshow(result.score_map, cmap='hot')
    axes[1].set_title('Laser likelihood score map')
    plt.colorbar(im1, ax=axes[1])

    # Top detection patch
    if result.detections:
        det = result.detections[0]
        pad = 30
        h, w = result.score_map.shape
        y0 = max(int(det.y) - pad, 0)
        y1 = min(int(det.y) + pad + 1, h)
        x0 = max(int(det.x) - pad, 0)
        x1 = min(int(det.x) + pad + 1, w)
        patch = result.score_map[y0:y1, x0:x1]

        im2 = axes[2].imshow(patch, cmap='hot')
        local_x = det.x - x0
        local_y = det.y - y0
        axes[2].plot(local_x, local_y, 'w+', ms=12, mew=2)
        theta = np.linspace(0, 2 * np.pi, 100)
        axes[2].plot(local_x + det.sigma_x * np.cos(theta),
                     local_y + det.sigma_y * np.sin(theta),
                     'w--', lw=1.5)
        axes[2].set_title(f'Top detection patch\n'
                          f'({det.x:.1f}, {det.y:.1f}) '
                          f'sigma=({det.sigma_x:.1f}, {det.sigma_y:.1f})\n'
                          f'conf={det.confidence:.3f}')
        plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()
    return fig
