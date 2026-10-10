import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median

import cv2 as cv
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classical_cv.classical_detector import detect_objects


DATASET_VAL = PROJECT_ROOT / "Dataset" / "images" / "val"
DATASET_LABEL = PROJECT_ROOT / "Dataset" / "labels" / "val"
MODEL_PATH = (
    PROJECT_ROOT / "deep_learning" / "runs" / "shape_color_yolo-3" / "weights" / "best.pt"
)

IOU_THRESHOLD = 0.5
YOLO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = "cpu"

CLASS_NAMES = {
    0: "red_circle",
    1: "red_square",
    2: "red_triangle",
    3: "blue_circle",
    4: "blue_square",
    5: "blue_triangle",
    6: "green_circle",
    7: "green_square",
    8: "green_triangle",
}


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    width = max(0, x_right - x_left)
    height = max(0, y_bottom - y_top)
    intersection = width * height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path, image_width, image_height):
    ground_truth = []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing annotation: {label_path}")

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if not values:
                continue

            class_id = int(values[0])
            cx, cy, w, h = map(float, values[1:5])
            ground_truth.append({
                "class_name": CLASS_NAMES[class_id],
                "bbox": (
                    int((cx - w / 2) * image_width),
                    int((cy - h / 2) * image_height),
                    int((cx + w / 2) * image_width),
                    int((cy + h / 2) * image_height),
                ),
            })

    return ground_truth


def count_matches(predictions, ground_truth):
    candidates = []

    for pred_idx, prediction in enumerate(predictions):
        for gt_idx, actual in enumerate(ground_truth):
            if prediction["class_name"] != actual["class_name"]:
                continue

            iou = calculate_iou(prediction["bbox"], actual["bbox"])
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(reverse=True)
    matched_predictions = set()
    matched_ground_truth = set()

    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_predictions or gt_idx in matched_ground_truth:
            continue
        matched_predictions.add(pred_idx)
        matched_ground_truth.add(gt_idx)

    tp = len(matched_predictions)
    return tp, len(predictions) - tp, len(ground_truth) - tp


def predict_yolo(model, image):
    result = model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        device=YOLO_DEVICE,
        verbose=False,
    )[0]

    predictions = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(result.names[class_id]).lower().replace("-", "_").replace(" ", "_")
        predictions.append({
            "class_name": class_name,
            "bbox": tuple(box.xyxy[0].tolist()),
        })

    return predictions


def summarize(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_seconds = sum(stats["times"])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_ms": mean(stats["times"]) * 1000,
        "median_ms": median(stats["times"]) * 1000,
        "fps": len(stats["times"]) / total_seconds if total_seconds else 0.0,
    }


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {MODEL_PATH}")

    image_paths = sorted(
        path for path in DATASET_VAL.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    samples = []
    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        height, width = image.shape[:2]
        ground_truth = load_ground_truth(
            DATASET_LABEL / f"{image_path.stem}.txt", width, height
        )
        samples.append((image, ground_truth))

    if not samples:
        raise ValueError(f"No validation images found in {DATASET_VAL}")

    model = YOLO(str(MODEL_PATH))

    for _ in range(3):
        detect_objects(samples[0][0])
        predict_yolo(model, samples[0][0])

    stats = {
        "Classical CV": {"tp": 0, "fp": 0, "fn": 0, "times": []},
        "YOLO11": {"tp": 0, "fp": 0, "fn": 0, "times": []},
    }

    for image, ground_truth in samples:
        start = time.perf_counter()
        classical_predictions = detect_objects(image)
        classical_seconds = time.perf_counter() - start

        start = time.perf_counter()
        yolo_predictions = predict_yolo(model, image)
        yolo_seconds = time.perf_counter() - start

        for name, predictions, duration in (
            ("Classical CV", classical_predictions, classical_seconds),
            ("YOLO11", yolo_predictions, yolo_seconds),
        ):
            tp, fp, fn = count_matches(predictions, ground_truth)
            stats[name]["tp"] += tp
            stats[name]["fp"] += fp
            stats[name]["fn"] += fn
            stats[name]["times"].append(duration)

    results = {name: summarize(data) for name, data in stats.items()}

    print(f"\nHardware: {platform.processor() or platform.machine()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"YOLO device: {YOLO_DEVICE}")
    print(f"Validation images: {len(samples)} | IoU threshold: {IOU_THRESHOLD}")
    print(f"YOLO confidence threshold: {YOLO_CONFIDENCE}")
    print("Disk I/O, label loading, model startup, and warmup are excluded from timing.")
    print(f"\n{'Metric':<24} {'Classical CV':>16} {'YOLO11':>16}")
    print("-" * 58)

    for label, key, suffix in (
        ("Precision", "precision", "%"),
        ("Recall", "recall", "%"),
        ("F1-score", "f1", "%"),
        ("Avg latency / image", "average_ms", " ms"),
        ("Median latency / image", "median_ms", " ms"),
        ("FPS", "fps", ""),
    ):
        a, b = results["Classical CV"][key], results["YOLO11"][key]
        if suffix == "%":
            a *= 100
            b *= 100
        print(f"{label:<24} {a:>14.2f}{suffix:<2} {b:>14.2f}{suffix:<2}")

    print("\nCounts:")
    for name, data in stats.items():
        print(f"{name}: TP={data['tp']}, FP={data['fp']}, FN={data['fn']}")

    classical_ms = results["Classical CV"]["average_ms"]
    yolo_ms = results["YOLO11"]["average_ms"]
    if classical_ms > 0 and yolo_ms > 0:
        faster = "Classical CV" if classical_ms < yolo_ms else "YOLO11"
        speedup = max(classical_ms, yolo_ms) / min(classical_ms, yolo_ms)
        print(f"\nFaster (average CPU latency): {faster} by {speedup:.2f}x")


if __name__ == "__main__":
    main()
