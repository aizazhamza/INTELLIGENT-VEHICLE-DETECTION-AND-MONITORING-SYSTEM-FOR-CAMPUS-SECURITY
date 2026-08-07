from __future__ import annotations

import argparse
import time
import cv2
import os
import re
import easyocr
from collections import Counter

from matplotlib.pylab import rint

import campus_db
import p1
import p4

reader = easyocr.Reader(["en"], gpu=False)


# =========================
# UTIL
# =========================
def clean_plate(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < 60


def is_valid_plate(text):
    return bool(text) and 5 <= len(text) <= 12


# =========================
# OCR
# =========================
def recognize_plate_easyocr(crop):

    if crop is None or crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=25)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    results = reader.readtext(thresh, detail=1, paragraph=False)

    def _extract_text_conf(item):
        if isinstance(item, dict):
            text = item.get("text", "")
            conf = item.get("confidence", item.get("conf", 0))
            return text, conf
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            return item[1], item[2]
        return "", 0

    candidates = []

    for item in results:
        text, conf = _extract_text_conf(item)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue

        if conf < 0.05:
            continue

        cleaned = clean_plate(str(text))

        if is_valid_plate(cleaned):
            candidates.append((cleaned, conf))

    if candidates:
        return max(candidates, key=lambda x: x[1])[0]

    if results:
        first_text, _ = _extract_text_conf(results[0])
        return clean_plate(str(first_text))

    return ""


# =========================
# CORE PIPELINE
# =========================
def process_camera_stream(frame, gate, tracks, last_saved,
                          cooldowns, frame_id, model, db_conn, args,
                          ocr_votes):

    H, W = frame.shape[:2]
    now = time.time()

    detections, _, _ = p1.detect_plates(frame, model, conf_threshold=args.conf)
    print("DETECTIONS:", len(detections))
    boxes = [d.bbox for d in detections]

    tracks = p4.TrackLicensePlates(
        boxes,
        tracks,
        association_threshold=args.assoc,
        max_age=args.max_age,
    )
    

    results = []

    for t in tracks:

        if t.hits < args.hits:
            continue

        if now - last_saved.get(t.id, 0) < 1.2:
            continue

        x1, y1, x2, y2 = map(int, t.bbox)

        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = int((x2 - x1) * 0.5)
        pad_y = int((y2 - y1) * 0.6)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or is_blurry(crop):
            continue

        plate = recognize_plate_easyocr(crop)
        print("OCR RESULT:", plate)
        if not plate:
            continue

        # OCR voting
        ocr_votes.setdefault(t.id, [])
        ocr_votes[t.id].append(plate)
        ocr_votes[t.id] = ocr_votes[t.id][-5:]

        if len(ocr_votes[t.id]) < 2:
            continue

        vote_counts = Counter(ocr_votes[t.id])
        best_plate, best_count = vote_counts.most_common(1)[0]

        if best_count < 2:
            continue

        plate = best_plate
        print("TRACK:", t.id, "OCR:", plate)
        print("VOTES:", ocr_votes.get(t.id))

        # attach to track (IMPORTANT FOR UI)
        t.last_plate = plate

        if plate in cooldowns and (now - cooldowns[plate] < args.cooldown):
            continue

        cooldowns[plate] = now

        event_type = campus_db.detect_event_type(db_conn, plate)

        os.makedirs("static/plates", exist_ok=True)
        img_path = f"static/plates/G{gate}_{plate}_{frame_id}.jpg"
        cv2.imwrite(img_path, crop)
        print("IMAGE SAVED:", img_path)
        print("DATABASE SAVE:", plate)

        campus_db.insert_event(
            db_conn,
            plate_number=plate,
            event_type=event_type,
            image_path=img_path,
            confidence=0.92,
            gate_number=gate
        )

        print("🔥 SAVED:", plate)

        results.append({
            "track_id": t.id,
            "plate_number": plate,
            "event_type": event_type,
            "image_path": img_path,
            "gate_number": gate
        })

        last_saved[t.id] = now

    return tracks, results


# =========================
# LIVE MODE
# =========================
def run_live_camera(cam_index, gate, model, db_conn, args):

    cap = cv2.VideoCapture(cam_index)
    cv2.namedWindow("ANPR SYSTEM", cv2.WINDOW_NORMAL)

    tracks = []
    last_saved = {}
    cooldowns = {}
    ocr_votes = {}
    frame_id = 0
    frame_skip = 1

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % frame_skip == 0:

            tracks, _ = process_camera_stream(
                frame, gate, tracks,
                last_saved, cooldowns,
                frame_id, model, db_conn, args,
                ocr_votes
            )

        for t in tracks:
            

            x1, y1, x2, y2 = map(int, t.bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            label = f"ID:{t.id}"
            if hasattr(t, "last_plate"):
                label += f" | {t.last_plate}"
                    
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)
            

        cv2.imshow("ANPR SYSTEM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break

    cap.release()
    cv2.destroyAllWindows()
    
    


# =========================
# VIDEO MODE
# =========================
def process_video_file(video_path, gate, model, db_conn, args):

    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("ANPR SYSTEM", cv2.WINDOW_NORMAL)

    tracks = []
    last_saved = {}
    cooldowns = {}
    ocr_votes = {}
    frame_id = 0
    
    frame_skip = 1

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % frame_skip == 0:
            tracks, _ = process_camera_stream(
                frame, gate, tracks,
                last_saved, cooldowns,
                frame_id, model, db_conn, args,
                ocr_votes
            )

        for t in tracks:

            x1, y1, x2, y2 = map(int, t.bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID:{t.id}"
            if hasattr(t, "last_plate"):
                label += f" | {t.last_plate}"

            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)
            

        cv2.imshow("ANPR SYSTEM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()
    


# =========================
# ARGS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", default="0")
    parser.add_argument("--gate", type=int, default=1)
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--assoc", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=80)
    parser.add_argument("--hits", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=6.0)
    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    args = parse_args()

    model = p1.load_yolo_model(args.model)
    db_conn = campus_db.connect_database()

    source = args.source1

    if str(source).endswith((".mp4", ".avi", ".mkv", ".mov")):
        process_video_file(source, args.gate, model, db_conn, args)
    else:
        run_live_camera(int(source), args.gate, model, db_conn, args)

    db_conn.close()