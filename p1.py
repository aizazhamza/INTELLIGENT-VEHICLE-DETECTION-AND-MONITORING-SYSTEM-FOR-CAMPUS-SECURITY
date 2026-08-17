"""
p1.py - YOLO License Plate Detection (SAFE VERSION)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_id: Optional[int] = None


# =========================
# MODEL LOAD
# =========================
def load_yolo_model(model_path: str = "best.pt"):

    if YOLO is None:
        raise ImportError("Install ultralytics: pip install ultralytics")

    return YOLO(model_path)


# =========================
# IMAGE ENHANCEMENT
# =========================
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return cv2.convertScaleAbs(blurred, alpha=1.2, beta=20)


# =========================
# SAFE BBOX CLIP
# =========================
def _clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    return x1, y1, x2, y2


# =========================
# DETECTION FUNCTION
# =========================
def detect_plates(
    frame,
    yolo_model,
    conf_threshold=0.5,
    classes: Optional[Sequence[int]] = None,
):

    enhanced = enhance_frame(frame)

    results = yolo_model(
    enhanced,
    imgsz=640,
    verbose=False
)

    detections = []
    crops = []

    if results is None or len(results) == 0:
        return detections, crops, enhanced

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections, crops, enhanced

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    cls = None
    try:
        if boxes.cls is not None:
            cls = boxes.cls.cpu().numpy().astype(int)
    except:
        cls = None

    class_set = set(classes) if classes is not None else None

    H, W = frame.shape[:2]

    for i, b in enumerate(xyxy):

        if float(conf[i]) < conf_threshold:
            continue

        class_id = int(cls[i]) if cls is not None else None

        if class_set is not None and class_id not in class_set:
            continue

        x1, y1, x2, y2 = _clip_bbox(b, W, H)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = crop.copy()

        detections.append(
            Detection(bbox=[x1, y1, x2, y2],
                      confidence=float(conf[i]),
                      class_id=class_id)
        )

        crops.append(crop)

    return detections, crops, enhanced


# =========================
# TEST MODE
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo_11.pt")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.35)

    args = parser.parse_args()

    model = load_yolo_model(args.model)

    source = args.source
    try:
        source = int(source)
    except:
        pass

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit("Cannot open source")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets, _, _ = detect_plates(frame, model, args.conf)

        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("ALPR", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()