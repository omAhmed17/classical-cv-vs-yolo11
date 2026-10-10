import sys
import time
import urllib.request
import cv2 as cv
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical_cv.classical_detector import detect_objects


app = FastAPI(
    title="Shape & Color Sorting API - Model Comparison",
    version="2.0.0"
)

MODEL_PATH = PROJECT_ROOT / "deep_learning" / "runs" / "shape_color_yolo-3" / "weights" / "best.pt"

# Replace this URL with your actual Hugging Face model URL after uploading best.pt
HF_MODEL_URL = "https://huggingface.co/omarerlasheedy/shape-color/resolve/main/best.pt"

if not MODEL_PATH.exists():
    print("Downloading YOLO model weights...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)
    print("Download complete.")

try:
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



@app.post("/predict/deep_learning", tags=["Models"])
async def predict_yolo(file: UploadFile = File(...)):
    """Predict using the YOLO11n deep learning model."""
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})
            
        start_time = time.perf_counter()
        results = model.predict(source=image, conf=0.25, verbose=False)
        end_time = time.perf_counter()
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                
        return {
            "filename": file.filename,
            "inference_time_ms": round((end_time - start_time) * 1000, 2),
            "total_objects_detected": len(detections),
            "detections": detections
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/classical", tags=["Models"])
async def predict_classical(file: UploadFile = File(...)):
    """Predict using Classical Computer Vision (OpenCV contours & HSV masking)."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})
            
        start_time = time.perf_counter()
        raw_detections = detect_objects(image)
        end_time = time.perf_counter()
        
        detections = []
        for det in raw_detections:
            detections.append({
                "class": det["class_name"],
                "color": det["color"],
                "shape": det["shape"],
                "bbox": det["bbox"]
            })
            
        return {
            "filename": file.filename,
            "inference_time_ms": round((end_time - start_time) * 1000, 2),
            "total_objects_detected": len(detections),
            "detections": detections
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/compare", tags=["Comparison"])
async def predict_compare(file: UploadFile = File(...)):
    """Run BOTH models on the same image to compare speed and results."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})
            
        start_c = time.perf_counter()
        raw_det_c = detect_objects(image)
        end_c = time.perf_counter()
        
        classical_results = []
        for det in raw_det_c:
            classical_results.append({
                "class": det["class_name"],
                "bbox": det["bbox"]
            })
            
        yolo_results = []
        end_y = start_y = 0
        if model:
            start_y = time.perf_counter()
            results_y = model.predict(source=image, conf=0.25, verbose=False)
            end_y = time.perf_counter()
            
            for r in results_y:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    yolo_results.append({
                        "class": class_name,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
        return {
            "filename": file.filename,
            "classical_cv": {
                "inference_time_ms": round((end_c - start_c) * 1000, 2),
                "total_detected": len(classical_results),
                "detections": classical_results
            },
            "deep_learning_yolo": {
                "inference_time_ms": round((end_y - start_y) * 1000, 2),
                "total_detected": len(yolo_results),
                "detections": yolo_results
            }
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
