from ultralytics import YOLO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "deep_learning"
    / "runs"
    / "shape_color_yolo-3"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "Dataset"
    / "data.yaml"
)


model = YOLO(str(MODEL_PATH))


metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640
)


precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map50_95 = metrics.box.map


if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0


print("\n==============================")
print("YOLO VALIDATION RESULTS")
print("==============================")

print(f"Precision:   {precision * 100:.2f}%")
print(f"Recall:      {recall * 100:.2f}%")
print(f"F1 Score:    {f1 * 100:.2f}%")
print(f"mAP@0.50:    {map50 * 100:.2f}%")
print(f"mAP@0.50:95: {map50_95 * 100:.2f}%")