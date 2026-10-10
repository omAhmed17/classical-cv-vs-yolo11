from ultralytics import YOLO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_ROOT / "Dataset" / "data.yaml"


model = YOLO("yolo11n.pt")


model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    project=str(PROJECT_ROOT / "deep_learning" / "runs"),
    name="shape_color_yolo"
)