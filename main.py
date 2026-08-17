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

def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < 60


def sharpness_score(img):
    if img is None or img.size == 0:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def is_valid_plate(text):
    """
    Basic ANPR plate validation.

    Rejects:
    - Empty text
    - Very short OCR results
    - Text containing no digits
    - Text containing no letters

    Accepts different Pakistani plate formats
    without forcing one exact format.
    """

    if not text:
        return False

    text = clean_plate(text)

    # Reasonable length
    if not 6 <= len(text) <= 12:
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in text):
        return False

    # Must contain at least one number
    if not any(c.isdigit() for c in text):
        return False

    return True


# =========================
# OCR
# =========================
def recognize_plate_easyocr(crop):
    """
    Improved EasyOCR for ANPR.

    Strategy:
    1. Resize the plate.
    2. Create multiple preprocessing versions.
    3. Run EasyOCR on each version.
    4. Collect valid OCR results.
    5. Use confidence + frequency to select the best result.
    """

    if crop is None or crop.size == 0:
        return ""

    try:
        # --------------------------------------------------
        # 1. Resize original plate
        # --------------------------------------------------
        resized = cv2.resize(
            crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        # --------------------------------------------------
        # 2. Grayscale
        # --------------------------------------------------
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------
        # 3. Contrast enhancement
        # --------------------------------------------------
        enhanced = cv2.convertScaleAbs(
            gray,
            alpha=1.8,
            beta=25
        )

        # --------------------------------------------------
        # 4. CLAHE
        # --------------------------------------------------
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        clahe_img = clahe.apply(enhanced)

        # --------------------------------------------------
        # 5. Light blur
        # --------------------------------------------------
        blurred = cv2.GaussianBlur(
            clahe_img,
            (3, 3),
            0
        )

        # --------------------------------------------------
        # 6. Adaptive threshold
        # --------------------------------------------------
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )

        # --------------------------------------------------
        # Create multiple OCR inputs
        # --------------------------------------------------
        ocr_images = [
            resized,
            gray,
            clahe_img,
            thresh
        ]

        candidates = []

        # --------------------------------------------------
        # 7. Run OCR on every version
        # --------------------------------------------------
        for img in ocr_images:

            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            )

            for item in results:

                text = ""
                conf = 0.0

                if isinstance(item, dict):

                    text = item.get("text", "")

                    conf = item.get(
                        "confidence",
                        item.get("conf", 0)
                    )

                elif isinstance(item, (list, tuple)):

                    if len(item) >= 3:
                        text = item[1]
                        conf = item[2]

                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    continue

                # Ignore extremely weak OCR results
                if conf < 0.20:
                    continue

                cleaned = clean_plate(str(text))

                if not cleaned:
                    continue

                if is_valid_plate(cleaned):

                    candidates.append(
                        (cleaned, conf)
                    )

        # --------------------------------------------------
        # 8. No valid OCR result
        # --------------------------------------------------
        if not candidates:
            return ""

        # --------------------------------------------------
        # 9. Voting
        # --------------------------------------------------
        votes = {}

        for text, conf in candidates:

            if text not in votes:
                votes[text] = {
                    "count": 0,
                    "confidence": 0.0
                }

            votes[text]["count"] += 1
            votes[text]["confidence"] += conf

        # --------------------------------------------------
        # 10. Select strongest candidate
        #
        # Priority:
        #       1. Number of votes
        #       2. Average confidence
        # --------------------------------------------------
        best_text = None
        best_score = None

        for text, data in votes.items():

            count = data["count"]

            avg_conf = (
                data["confidence"] / count
            )

            score = (
                count * 0.7
                +
                avg_conf * 0.3
            )

            if best_score is None or score > best_score:

                best_score = score
                best_text = text

        return best_text if best_text else ""

    except Exception as e:

        print(
            f"[OCR ERROR] {e}"
        )

        return ""# =========================
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
    parser.add_argument("--model", default="yolo_8.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--assoc", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=80)
    parser.add_argument("--hits", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=60.0)
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