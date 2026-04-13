"""Evaluate the laser detection pipeline against a simple point-label manifest."""

import argparse
import json
from pathlib import Path
from statistics import median

from pipeline import detect_laser


DEFAULT_LABELS_PATH = Path("eval/labels.json")
DEFAULT_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _load_manifest(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "images" in data:
        entries = data["images"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Expected a list of entries or an {'images': [...]} object")

    normalized = []
    for entry in entries:
        points = entry.get("points", [])
        normalized.append({
            "image": entry["image"],
            "points": [
                {"x": float(point["x"]), "y": float(point["y"])}
                for point in points
            ],
        })
    return normalized


def _resolve_image_path(repo_root, image_ref):
    image_path = Path(image_ref)
    if image_path.is_absolute():
        return image_path
    return repo_root / image_path


def _nearest_unmatched_point(detection, points, matched_points, radius):
    best_index = None
    best_distance = None
    for index, point in enumerate(points):
        if index in matched_points:
            continue
        distance = ((detection.x - point["x"]) ** 2 + (detection.y - point["y"]) ** 2) ** 0.5
        if distance <= radius and (best_distance is None or distance < best_distance):
            best_index = index
            best_distance = distance
    return best_index, best_distance


def evaluate_manifest(entries, repo_root, min_confidence, radius):
    total_images = len(entries)
    hit_count = 0
    matched_top1_errors = []
    total_false_positives = 0

    for entry in entries:
        image_path = _resolve_image_path(repo_root, entry["image"])
        points = entry["points"]
        result = detect_laser(
            image_path=str(image_path),
            min_confidence=min_confidence,
            max_detections=max(10, len(points) * 3),
        )

        detections = result.detections
        if points and detections:
            distances = [
                ((detections[0].x - point["x"]) ** 2 + (detections[0].y - point["y"]) ** 2) ** 0.5
                for point in points
            ]
            top1_error = min(distances)
            if top1_error <= radius:
                hit_count += 1
                matched_top1_errors.append(top1_error)

        matched_points = set()
        for detection in detections:
            match_index, _ = _nearest_unmatched_point(detection, points, matched_points, radius)
            if match_index is None:
                total_false_positives += 1
            else:
                matched_points.add(match_index)

    return {
        "threshold": float(min_confidence),
        "total_images": total_images,
        "top1_hit_rate": (hit_count / total_images) if total_images else 0.0,
        "median_localization_error": (
            float(median(matched_top1_errors)) if matched_top1_errors else None
        ),
        "false_positives_per_image": (
            total_false_positives / total_images if total_images else 0.0
        ),
    }


def choose_threshold(results, fallback_threshold):
    baseline = next((result for result in results
                     if abs(result["threshold"] - fallback_threshold) < 1e-9), None)
    if baseline is None:
        baseline = min(results, key=lambda result: abs(result["threshold"] - fallback_threshold))

    eligible = [
        result for result in results
        if result["top1_hit_rate"] >= baseline["top1_hit_rate"] and
        result["false_positives_per_image"] <= baseline["false_positives_per_image"]
    ]
    if not eligible:
        return fallback_threshold

    eligible.sort(
        key=lambda result: (
            result["false_positives_per_image"],
            -result["top1_hit_rate"],
            result["median_localization_error"]
            if result["median_localization_error"] is not None else float("inf"),
            abs(result["threshold"] - fallback_threshold),
        )
    )
    return eligible[0]["threshold"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH,
                        help="Path to eval/labels.json")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Matching radius in pixels for hits and false positives")
    parser.add_argument("--fallback-threshold", type=float, default=0.35,
                        help="Fallback min_confidence if the sweep finds no improvement")
    parser.add_argument("--sweep", nargs="*", type=float, default=DEFAULT_SWEEP,
                        help="Confidence thresholds to evaluate")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    labels_path = args.labels if args.labels.is_absolute() else repo_root / args.labels
    if not labels_path.exists():
        raise FileNotFoundError(f"Label manifest not found: {labels_path}")

    entries = _load_manifest(labels_path)
    if not entries:
        print(f"No labels found in {labels_path}. Populate the manifest and rerun.")
        return

    thresholds = sorted(set(args.sweep + [args.fallback_threshold]))
    results = [
        evaluate_manifest(entries, repo_root, threshold, args.radius)
        for threshold in thresholds
    ]

    print("threshold\ttop1_hit_rate\tmedian_error\tfalse_positives_per_image")
    for result in results:
        median_error = (
            f"{result['median_localization_error']:.3f}"
            if result["median_localization_error"] is not None else "n/a"
        )
        print(
            f"{result['threshold']:.2f}\t"
            f"{result['top1_hit_rate']:.3f}\t"
            f"{median_error}\t"
            f"{result['false_positives_per_image']:.3f}"
        )

    recommended = choose_threshold(results, args.fallback_threshold)
    print(f"\nRecommended min_confidence: {recommended:.2f}")


if __name__ == "__main__":
    main()
