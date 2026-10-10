from classical_detector import detect_objects
import cv2 as cv
from pathlib import Path


ROOT_FOLDER = Path(__file__).resolve().parents[1]
DATASET_VAL = ROOT_FOLDER / "Dataset" / "images" / "val"
DATASET_LABEL = ROOT_FOLDER / "Dataset" / "labels" / "val"

IOU_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "red_circle",
    1: "red_square",
    2: "red_triangle",
    3: "blue_circle",
    4: "blue_square",
    5: "blue_triangle",
    6: "green_circle",
    7: "green_square",
    8: "green_triangle"
}


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_ground_truth(label_path, image_width, image_height):
    ground_truth_boxes = []

    if not label_path.exists():
        return ground_truth_boxes

    with open(label_path, "r") as file:
        labels = file.readlines()

    for line in labels:
        values = line.strip().split()

        if not values:
            continue

        class_id = int(values[0])
        center_x = float(values[1])
        center_y = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        x1 = int((center_x - box_width / 2) * image_width)
        y1 = int((center_y - box_height / 2) * image_height)
        x2 = int((center_x + box_width / 2) * image_width)
        y2 = int((center_y + box_height / 2) * image_height)

        ground_truth_boxes.append({
            "class_name": CLASS_NAMES[class_id],
            "bbox": (x1, y1, x2, y2)
        })

    return ground_truth_boxes


def evaluate_detections(detections, ground_truth_boxes):
    possible_matches = []

    for pred_index, prediction in enumerate(detections):
        for gt_index, actual in enumerate(ground_truth_boxes):

            if prediction["class_name"] != actual["class_name"]:
                continue

            iou = calculate_iou(
                prediction["bbox"],
                actual["bbox"]
            )

            if iou >= IOU_THRESHOLD:
                possible_matches.append((iou, pred_index, gt_index))

    possible_matches.sort(reverse=True)

    matched_predictions = set()
    matched_ground_truths = set()

    for iou, pred_index, gt_index in possible_matches:
        if pred_index in matched_predictions:
            continue

        if gt_index in matched_ground_truths:
            continue

        matched_predictions.add(pred_index)
        matched_ground_truths.add(gt_index)

    tp = len(matched_predictions)
    fp = len(detections) - tp
    fn = len(ground_truth_boxes) - tp

    return tp, fp, fn


def main():
    image_paths = sorted(
        path for path in DATASET_VAL.iterdir()
        if path.suffix.lower() in [".jpg", ".png", ".jpeg"]
    )

    total_images = 0
    total_detections = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_path in image_paths:
        image = cv.imread(str(image_path))

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        detections = detect_objects(image)

        image_height, image_width = image.shape[:2]
        label_path = DATASET_LABEL / f"{image_path.stem}.txt"

        ground_truth_boxes = load_ground_truth(
            label_path,
            image_width,
            image_height
        )

        tp, fp, fn = evaluate_detections(
            detections,
            ground_truth_boxes
        )

        total_images += 1
        total_detections += len(detections)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"{image_path.name}")
        print(f"Predictions: {len(detections)}")
        print(f"Actual objects: {len(ground_truth_boxes)}")
        print()

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0 else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    print("=" * 42)
    print("  CLASSICAL CV - VALIDATION RESULTS")
    print("=" * 42)
    print(f"  Total images     : {total_images}")
    print(f"  Total detections : {total_detections}")
    print("-" * 42)
    print(f"  True Positives   : {total_tp}")
    print(f"  False Positives  : {total_fp}")
    print(f"  False Negatives  : {total_fn}")
    print("-" * 42)
    print(f"  Precision        : {precision * 100:.2f}%")
    print(f"  Recall           : {recall * 100:.2f}%")
    print(f"  F1 Score         : {f1 * 100:.2f}%")
    print("=" * 42)
    print(f"  IoU Threshold    : {IOU_THRESHOLD}")


if __name__ == "__main__":
    main()