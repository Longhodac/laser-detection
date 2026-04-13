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
from detectors import Candidate, run_detectors
from fgf_full import fgf_full


DEFAULT_DETECTOR_THRESHOLDS = {
    "log_threshold": 0.15,
    "dog_threshold": 0.15,
    "tophat_threshold_factor": 4.0,
    "radial_threshold_factor": 4.0,
}


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
    fit_source: Optional[str] = None
    candidate_response: float = 0.0
    detector_support: int = 0
    candidate_drift: float = 0.0


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


def _compute_response_conf(score_map, cx, cy, inner_radius=2, outer_radius=6):
    """Compute how strongly the candidate stands out in the score map."""
    h, w = score_map.shape
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    y0 = max(cy_i - outer_radius, 0)
    y1 = min(cy_i + outer_radius + 1, h)
    x0 = max(cx_i - outer_radius, 0)
    x1 = min(cx_i + outer_radius + 1, w)
    patch = score_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    peak = float(score_map[cy_i, cx_i])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = patch[(dist >= inner_radius) & (dist <= outer_radius)]
    if ring.size == 0:
        baseline = float(np.median(patch))
    else:
        baseline = float(np.percentile(ring, 90))

    local_prominence = max(0.0, peak - baseline)
    return float(np.clip(local_prominence + 0.1 * peak, 0.0, 1.0))


def _size_score(sigma_x, sigma_y):
    """Soft size sanity for fitted laser blobs."""
    max_sigma = max(sigma_x, sigma_y)
    min_sigma = min(sigma_x, sigma_y)

    if max_sigma <= 5.0:
        score = 1.0
    elif max_sigma <= 8.0:
        score = 0.6
    else:
        score = 0.2

    if min_sigma < 0.6:
        score *= 0.5

    return float(score)


def _aspect_score(sigma_x, sigma_y):
    """Soft aspect-ratio sanity for fitted laser blobs."""
    aspect = max(sigma_x, sigma_y) / (min(sigma_x, sigma_y) + 1e-8)
    if aspect <= 2.5:
        return 1.0
    if aspect <= 4.0:
        return 0.5
    return 0.1


def _evaluate_fit(source_name, fit_result, expected_x, expected_y, pad):
    """Score and validate one FGF fit candidate."""
    xc, yc, sx, sy, amplitude, raw_conf = fit_result
    values = np.array([xc, yc, sx, sy, amplitude, raw_conf], dtype=np.float64)
    if not np.all(np.isfinite(values)) or sx <= 0 or sy <= 0 or amplitude <= 0:
        return None

    drift = float(np.hypot(xc - expected_x, yc - expected_y))
    max_sigma = max(sx, sy)
    aspect = max_sigma / (min(sx, sy) + 1e-8)
    hard_drift_limit = min(10.0, pad / 3.0)
    hard_reject = (
        drift > hard_drift_limit or
        max_sigma > (pad / 2.0) or
        aspect > 8.0
    )

    drift_score = float(np.clip(1.0 - drift / 10.0, 0.0, 1.0))
    size_score = _size_score(sx, sy)
    aspect_score = _aspect_score(sx, sy)
    sanity_conf = float(np.clip(
        0.5 * drift_score + 0.25 * size_score + 0.25 * aspect_score,
        0.0,
        1.0,
    ))
    fit_conf = float(np.clip(raw_conf * size_score * aspect_score, 0.0, 1.0))
    selection_score = 0.7 * fit_conf + 0.3 * sanity_conf

    return {
        "source": source_name,
        "xc": float(xc),
        "yc": float(yc),
        "sigma_x": float(sx),
        "sigma_y": float(sy),
        "amplitude": float(amplitude),
        "raw_conf": float(raw_conf),
        "fit_conf": fit_conf,
        "sanity_conf": sanity_conf,
        "selection_score": float(selection_score),
        "drift": drift,
        "drift_score": drift_score,
        "size_score": size_score,
        "aspect_score": aspect_score,
        "hard_reject": hard_reject,
    }


def _select_best_fit(score_fit, gray_fit):
    """Prefer stable score-map fits when grayscale is only marginally better."""
    valid_fits = [
        fit for fit in (score_fit, gray_fit)
        if fit is not None and not fit["hard_reject"]
    ]
    if not valid_fits:
        return None

    if score_fit is not None and gray_fit is not None:
        if not score_fit["hard_reject"] and not gray_fit["hard_reject"]:
            gray_margin = gray_fit["selection_score"] - score_fit["selection_score"]
            if gray_margin < 0.05 and gray_fit["sanity_conf"] < score_fit["sanity_conf"]:
                return score_fit

    return max(
        valid_fits,
        key=lambda fit: (fit["selection_score"], fit["sanity_conf"], fit["fit_conf"]),
    )


def detect_laser(image_path=None, image_bgr=None, laser_color="auto",
                 do_preprocess=True, pad=30, min_confidence=0.35,
                 max_detections=10, debug=False, detector_thresholds=None):
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
    detector_config = dict(DEFAULT_DETECTOR_THRESHOLDS)
    if detector_thresholds:
        detector_config.update(detector_thresholds)

    # --- Stage 1: Preprocess ---
    if do_preprocess:
        bgr_proc = preprocess(bgr)
    else:
        bgr_proc = bgr.copy()
    result.preprocessed = bgr_proc

    # --- Stage 2: Color scoring ---
    if debug:
        score_map, detected_color, score_debug = compute_laser_score(
            bgr_proc,
            color=laser_color,
            return_debug=True,
        )
        result.debug.update(score_debug)
    else:
        score_map, detected_color = compute_laser_score(
            bgr_proc,
            color=laser_color,
        )
    result.laser_color = detected_color
    result.score_map = score_map

    # --- Stage 3: Candidate detection ---
    gray = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pre_fgf_limit = 8 if max_detections <= 10 else min(max(max_detections * 2, 8), 32)

    candidates, det_debug = run_detectors(
        score_map, gray,
        use_log=True, use_tophat=True, use_radial=True,
        merge_radius=max(8, pad // 3),
        max_candidates=pre_fgf_limit,
        **detector_config,
    )

    if debug:
        result.debug.update(det_debug)
        result.debug["pre_fgf_limit"] = int(pre_fgf_limit)
        result.debug["min_confidence"] = float(min_confidence)

    # Always include score-map argmax as a candidate
    cy_peak, cx_peak = np.unravel_index(np.argmax(score_map), score_map.shape)
    peak_val = score_map[cy_peak, cx_peak]
    argmax_cand = Candidate(
        y=float(cy_peak),
        x=float(cx_peak),
        sigma=2.0,
        response=float(peak_val),
        detector_sources=("argmax",),
        detector_support=1,
    )

    if not candidates:
        candidates = [argmax_cand]
    else:
        # Add argmax if not already near an existing candidate
        already_near = any(
            np.sqrt((c.y - argmax_cand.y) ** 2 + (c.x - argmax_cand.x) ** 2) < 15
            for c in candidates
        )
        if not already_near:
            candidates.insert(0, argmax_cand)

    # --- Stage 4: FGF refinement on each candidate ---
    # Run FGF on both score_map and raw grayscale, keep the better fit
    detections = []
    for candidate in candidates:
        cy, cx, sigma_est, response = candidate
        patch_score, x0, y0 = extract_patch(score_map, cx, cy, pad=pad)
        patch_gray, _, _ = extract_patch(gray, cx, cy, pad=pad)
        if patch_score.shape[0] < 5 or patch_score.shape[1] < 5:
            continue

        expected_x = float(cx - x0)
        expected_y = float(cy - y0)
        score_fit = None
        gray_fit = None

        try:
            score_fit = _evaluate_fit(
                "score",
                fgf_full(patch_score.astype(np.float64)),
                expected_x,
                expected_y,
                pad,
            )
        except Exception:
            score_fit = None

        try:
            gray_fit = _evaluate_fit(
                "gray",
                fgf_full(patch_gray.astype(np.float64)),
                expected_x,
                expected_y,
                pad,
            )
        except Exception:
            gray_fit = None

        selected_fit = _select_best_fit(score_fit, gray_fit)
        if selected_fit is None:
            continue

        fit_conf = selected_fit["fit_conf"]
        response_conf = _compute_response_conf(score_map, cx, cy)
        sanity_conf = selected_fit["sanity_conf"]
        agreement_conf = min(1.0, candidate.detector_support / 3.0)
        final_conf = (
            0.45 * fit_conf +
            0.25 * response_conf +
            0.20 * sanity_conf +
            0.10 * agreement_conf
        )

        det = LaserDetection(
            x=x0 + selected_fit["xc"],
            y=y0 + selected_fit["yc"],
            sigma_x=selected_fit["sigma_x"],
            sigma_y=selected_fit["sigma_y"],
            amplitude=selected_fit["amplitude"],
            confidence=float(np.clip(final_conf, 0.0, 1.0)),
            patch_x0=x0,
            patch_y0=y0,
            fit_source=selected_fit["source"],
            candidate_response=response_conf,
            detector_support=candidate.detector_support,
            candidate_drift=selected_fit["drift"],
        )
        detections.append(det)

    # --- Filter, deduplicate, and sort ---
    detections = [d for d in detections if d.confidence >= min_confidence]
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Post-FGF deduplication: remove detections within merge_dist of a
    # higher-confidence detection
    merge_dist = max(8, pad // 3)
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
