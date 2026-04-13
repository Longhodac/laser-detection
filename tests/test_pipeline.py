import unittest

import cv2
import numpy as np

from color_scoring import compute_laser_score, detect_laser_color
from pipeline import detect_laser


def _make_background(shape=(128, 128), seed=0):
    """Build a mildly textured underwater-like background."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]

    b = 35 + 8 * (yy / max(h - 1, 1)) + 4 * np.sin(xx / 13.0)
    g = 24 + 5 * (xx / max(w - 1, 1)) + 3 * np.cos(yy / 17.0)
    r = 12 + 3 * np.sin((xx + yy) / 19.0)
    base = np.stack([b, g, r], axis=-1)
    noise = rng.normal(0.0, 3.0, size=(h, w, 3))
    image = np.clip(base + noise, 0, 255)
    return image.astype(np.uint8)


def _add_gaussian_spot(image, center, sigma, delta_bgr):
    """Add a colored Gaussian spot to an image."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    cx, cy = center
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))

    out = image.astype(np.float32).copy()
    for channel, delta in enumerate(delta_bgr):
        out[:, :, channel] += delta * gaussian
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_reflection_streak(image, center, axes, angle, delta_bgr):
    """Add a blurred elongated reflection-like streak."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        axes,
        angle,
        0,
        360,
        255,
        -1,
    )
    streak = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5.0, 1.5)

    out = image.astype(np.float32).copy()
    for channel, delta in enumerate(delta_bgr):
        out[:, :, channel] += delta * streak
    return np.clip(out, 0, 255).astype(np.uint8)


class PipelineSyntheticTests(unittest.TestCase):
    def test_centered_gaussian_dot_is_top_detection(self):
        image = _make_background(seed=1)
        image = _add_gaussian_spot(image, center=(64.4, 60.8), sigma=1.6,
                                   delta_bgr=(0, 0, 235))
        image = _add_gaussian_spot(image, center=(33.0, 94.0), sigma=3.2,
                                   delta_bgr=(0, 0, 120))
        image = _add_gaussian_spot(image, center=(98.0, 26.0), sigma=2.3,
                                   delta_bgr=(140, 140, 140))

        result = detect_laser(
            image_bgr=image,
            laser_color="red",
            do_preprocess=False,
            min_confidence=0.0,
            max_detections=5,
        )

        self.assertTrue(result.detections, "Expected at least one detection")
        top = result.detections[0]
        self.assertLess(np.hypot(top.x - 64.4, top.y - 60.8), 1.0)
        self.assertGreater(top.confidence, 0.35)
        if len(result.detections) > 1:
            self.assertGreaterEqual(top.confidence, result.detections[1].confidence)

    def test_edge_dot_confidence_remains_high(self):
        centered = _make_background(seed=2)
        centered = _add_gaussian_spot(centered, center=(64.0, 64.0), sigma=1.8,
                                      delta_bgr=(0, 0, 235))
        edge = _make_background(seed=2)
        edge = _add_gaussian_spot(edge, center=(5.5, 6.0), sigma=1.8,
                                  delta_bgr=(0, 0, 235))

        centered_result = detect_laser(
            image_bgr=centered,
            laser_color="red",
            do_preprocess=False,
            min_confidence=0.0,
            max_detections=3,
        )
        edge_result = detect_laser(
            image_bgr=edge,
            laser_color="red",
            do_preprocess=False,
            min_confidence=0.0,
            max_detections=3,
        )

        self.assertTrue(centered_result.detections)
        self.assertTrue(edge_result.detections)

        centered_top = centered_result.detections[0]
        edge_top = edge_result.detections[0]
        self.assertLess(np.hypot(centered_top.x - 64.0, centered_top.y - 64.0), 1.0)
        self.assertLess(np.hypot(edge_top.x - 5.5, edge_top.y - 6.0), 1.5)
        self.assertGreater(edge_top.confidence, 0.35)
        self.assertLess(abs(centered_top.confidence - edge_top.confidence), 0.15)

    def test_elongated_reflection_scores_below_acceptance_threshold(self):
        image = _make_background(seed=3)
        image = _add_reflection_streak(
            image,
            center=(64.0, 62.0),
            axes=(30, 4),
            angle=22,
            delta_bgr=(0, 0, 220),
        )

        result = detect_laser(
            image_bgr=image,
            laser_color="red",
            do_preprocess=False,
            min_confidence=0.0,
            max_detections=5,
        )

        if result.detections:
            self.assertLess(result.detections[0].confidence, 0.35)

    def test_auto_color_ignores_single_pixel_outlier(self):
        image = _make_background(seed=4)
        image = _add_gaussian_spot(image, center=(76.0, 52.0), sigma=1.7,
                                   delta_bgr=(0, 235, 0))
        image[0, 0] = np.array([0, 0, 255], dtype=np.uint8)

        self.assertEqual(detect_laser_color(image), "green")
        score_map, detected_color = compute_laser_score(image, color="auto")
        self.assertEqual(detected_color, "green")
        self.assertEqual(score_map.dtype, np.float32)
        self.assertGreater(float(score_map.max()), 0.5)


if __name__ == "__main__":
    unittest.main()
