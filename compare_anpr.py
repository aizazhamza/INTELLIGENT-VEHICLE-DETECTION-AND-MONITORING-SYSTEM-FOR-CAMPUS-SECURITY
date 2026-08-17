# import os
# import re
# import time
# import csv
# import gc

# import cv2
# import easyocr
# from ultralytics import YOLO


# # ============================================================
# # SETTINGS
# # ============================================================

# IMAGE_FOLDER = r"C:\Users\hamza\Downloads\archive (1)\images\images"

# YOLOV8_MODEL = r"E:\anpr_project\yolo_8.pt"
# YOLO11_MODEL = r"E:\anpr_project\yolo_11.pt"

# OUTPUT_FILE = r"E:\anpr_project\anpr_comparison.csv"

# CONF = 0.40


# # ============================================================
# # EASY OCR
# # ============================================================

# print("Loading EasyOCR...")

# reader = easyocr.Reader(
#     ["en"],
#     gpu=False,
#     verbose=False
# )

# print("EasyOCR loaded.")


# # ============================================================
# # CLEAN TEXT
# # ============================================================

# def clean_plate(text):

#     return re.sub(
#         r"[^A-Z0-9]",
#         "",
#         str(text).upper()
#     )


# def is_valid_plate(text):

#     return (
#         bool(text)
#         and 5 <= len(text) <= 12
#     )


# # ============================================================
# # OCR RECOGNITION ONLY
# # ============================================================

# def recognize_plate_easyocr(crop):

#     if crop is None or crop.size == 0:
#         return ""

#     try:

#         # -----------------------------------------
#         # Grayscale
#         # -----------------------------------------

#         gray = cv2.cvtColor(
#             crop,
#             cv2.COLOR_BGR2GRAY
#         )

#         # -----------------------------------------
#         # Resize only 2x
#         # -----------------------------------------

#         gray = cv2.resize(
#             gray,
#             None,
#             fx=2,
#             fy=2,
#             interpolation=cv2.INTER_CUBIC
#         )

#         # -----------------------------------------
#         # Contrast
#         # -----------------------------------------

#         gray = cv2.convertScaleAbs(
#             gray,
#             alpha=1.5,
#             beta=15
#         )

#         # -----------------------------------------
#         # Threshold
#         # -----------------------------------------

#         thresh = cv2.adaptiveThreshold(
#             gray,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             21,
#             5
#         )

#         # -----------------------------------------
#         # RECOGNITION ONLY
#         #
#         # detector=False prevents EasyOCR
#         # CRAFT detector from processing the crop.
#         # -----------------------------------------

#         results = reader.recognize(
#             thresh,
#             detail=1,
#             allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
#             decoder="greedy",
#             batch_size=1,
#             workers=0
#         )

#         candidates = []

#         for item in results:

#             if not isinstance(
#                 item,
#                 (list, tuple)
#             ):
#                 continue

#             if len(item) < 3:
#                 continue

#             text = item[1]
#             confidence = item[2]

#             try:
#                 confidence = float(
#                     confidence
#                 )
#             except:
#                 continue

#             cleaned = clean_plate(
#                 text
#             )

#             if (
#                 confidence >= 0.05
#                 and is_valid_plate(cleaned)
#             ):

#                 candidates.append(
#                     (
#                         cleaned,
#                         confidence
#                     )
#                 )

#         if candidates:

#             return max(
#                 candidates,
#                 key=lambda x: x[1]
#             )[0]

#         return ""

#     except Exception as e:

#         print(
#             "OCR ERROR:",
#             e
#         )

#         return ""


# # ============================================================
# # YOLO + OCR
# # ============================================================

# def detect_and_ocr(
#     model,
#     image,
#     model_name,
#     filename
# ):

#     start = time.perf_counter()

#     try:

#         prediction = model.predict(
#             source=image,
#             conf=CONF,
#             verbose=False,
#             device="cpu"
#         )[0]

#         best_box = None
#         best_conf = 0.0

#         if prediction.boxes is not None:

#             for box in prediction.boxes:

#                 confidence = float(
#                     box.conf[0]
#                 )

#                 if confidence > best_conf:

#                     best_conf = confidence

#                     best_box = (
#                         box.xyxy[0]
#                         .cpu()
#                         .numpy()
#                     )

#         # ------------------------------------------------
#         # No detection
#         # ------------------------------------------------

#         if best_box is None:

#             elapsed = (
#                 time.perf_counter()
#                 - start
#             ) * 1000

#             return {
#                 "image": filename,
#                 "model": model_name,
#                 "plate": "",
#                 "detection_confidence": 0,
#                 "time_ms": elapsed,
#                 "detected": 0,
#                 "ocr_success": 0
#             }

#         h, w = image.shape[:2]

#         x1, y1, x2, y2 = map(
#             int,
#             best_box
#         )

#         # ------------------------------------------------
#         # Small padding
#         # ------------------------------------------------

#         pad_x = int(
#             (x2 - x1) * 0.15
#         )

#         pad_y = int(
#             (y2 - y1) * 0.20
#         )

#         x1 = max(
#             0,
#             x1 - pad_x
#         )

#         y1 = max(
#             0,
#             y1 - pad_y
#         )

#         x2 = min(
#             w,
#             x2 + pad_x
#         )

#         y2 = min(
#             h,
#             y2 + pad_y
#         )

#         crop = image[
#             y1:y2,
#             x1:x2
#         ]

#         # ------------------------------------------------
#         # OCR
#         # ------------------------------------------------

#         plate = recognize_plate_easyocr(
#             crop
#         )

#         elapsed = (
#             time.perf_counter()
#             - start
#         ) * 1000

#         return {
#             "image": filename,
#             "model": model_name,
#             "plate": plate,
#             "detection_confidence": best_conf,
#             "time_ms": elapsed,
#             "detected": 1,
#             "ocr_success": 1 if plate else 0
#         }

#     except Exception as e:

#         print(
#             f"ERROR {filename}: {e}"
#         )

#         elapsed = (
#             time.perf_counter()
#             - start
#         ) * 1000

#         return {
#             "image": filename,
#             "model": model_name,
#             "plate": "",
#             "detection_confidence": 0,
#             "time_ms": elapsed,
#             "detected": 0,
#             "ocr_success": 0
#         }


# # ============================================================
# # TEST MODEL
# # ============================================================

# def test_model(
#     model_path,
#     model_name,
#     image_files
# ):

#     print()
#     print("=" * 70)
#     print(model_name)
#     print("=" * 70)

#     print(
#         "Loading model:",
#         model_path
#     )

#     model = YOLO(model_path)

#     results = []

#     detected = 0
#     ocr_success = 0

#     total_time = 0

#     for i, filename in enumerate(
#         image_files,
#         1
#     ):

#         path = os.path.join(
#             IMAGE_FOLDER,
#             filename
#         )

#         image = cv2.imread(path)

#         if image is None:

#             print(
#                 f"{i}/{len(image_files)} "
#                 f"{filename} | IMAGE ERROR"
#             )

#             continue

#         result = detect_and_ocr(
#             model,
#             image,
#             model_name,
#             filename
#         )

#         results.append(result)

#         if result["detected"]:
#             detected += 1

#         if result["ocr_success"]:
#             ocr_success += 1

#         total_time += result["time_ms"]

#         print(
#             f"{i}/{len(image_files)} "
#             f"{filename} | "
#             f"Plate: "
#             f"{result['plate'] or 'NONE'} | "
#             f"Conf: "
#             f"{result['detection_confidence']:.3f} | "
#             f"Time: "
#             f"{result['time_ms']:.1f} ms"
#         )

#         del image

#         if i % 5 == 0:

#             gc.collect()

#     # --------------------------------------------------------
#     # Summary
#     # --------------------------------------------------------

#     count = len(results)

#     avg_time = (
#         total_time / count
#         if count
#         else 0
#     )

#     fps = (
#         1000 / avg_time
#         if avg_time
#         else 0
#     )

#     print()
#     print("-" * 70)
#     print(model_name)
#     print("-" * 70)

#     print(
#         "Images:",
#         count
#     )

#     print(
#         "Plates detected:",
#         detected
#     )

#     print(
#         "OCR outputs:",
#         ocr_success
#     )

#     if count:

#         print(
#             f"Detection rate: "
#             f"{detected / count * 100:.2f}%"
#         )

#         print(
#             f"OCR output rate: "
#             f"{ocr_success / count * 100:.2f}%"
#         )

#     print(
#         f"Average time: "
#         f"{avg_time:.2f} ms"
#     )

#     print(
#         f"Approx FPS: "
#         f"{fps:.2f}"
#     )

#     # Free model memory

#     del model

#     gc.collect()

#     return results


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 70)
#     print("YOLOv8 + EasyOCR VS YOLO11 + EasyOCR")
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Images
#     # --------------------------------------------------------

#     image_files = sorted([
#         f
#         for f in os.listdir(
#             IMAGE_FOLDER
#         )
#         if f.lower().endswith(
#             (
#                 ".jpg",
#                 ".jpeg",
#                 ".png",
#                 ".bmp",
#                 ".webp"
#             )
#         )
#     ])

#     print()
#     print(
#         "Images found:",
#         len(image_files)
#     )

#     if not image_files:

#         print(
#             "No images found!"
#         )

#         raise SystemExit

#     # --------------------------------------------------------
#     # YOLOv8
#     # --------------------------------------------------------

#     v8_results = test_model(
#         YOLOV8_MODEL,
#         "YOLOv8 + EasyOCR",
#         image_files
#     )

#     # --------------------------------------------------------
#     # YOLO11
#     # --------------------------------------------------------

#     y11_results = test_model(
#         YOLO11_MODEL,
#         "YOLO11 + EasyOCR",
#         image_files
#     )

#     # --------------------------------------------------------
#     # Save CSV
#     # --------------------------------------------------------

#     all_results = (
#         v8_results +
#         y11_results
#     )

#     with open(
#         OUTPUT_FILE,
#         "w",
#         newline="",
#         encoding="utf-8"
#     ) as f:

#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "image",
#                 "model",
#                 "plate",
#                 "detection_confidence",
#                 "time_ms",
#                 "detected",
#                 "ocr_success"
#             ]
#         )

#         writer.writeheader()

#         writer.writerows(
#             all_results
#         )

#     # ========================================================
#     # FINAL TABLE
#     # ========================================================

#     def summary(results):

#         n = len(results)

#         if n == 0:
#             return (0, 0, 0, 0, 0)

#         detected = sum(
#             x["detected"]
#             for x in results
#         )

#         ocr = sum(
#             x["ocr_success"]
#             for x in results
#         )

#         avg = sum(
#             x["time_ms"]
#             for x in results
#         ) / n

#         fps = (
#             1000 / avg
#             if avg > 0
#             else 0
#         )

#         return (
#             n,
#             detected,
#             ocr,
#             avg,
#             fps
#         )

#     v8 = summary(
#         v8_results
#     )

#     y11 = summary(
#         y11_results
#     )

#     print()
#     print("=" * 75)
#     print(
#         "                    FINAL COMPARISON"
#     )
#     print("=" * 75)

#     print(
#         f"{'Metric':<30}"
#         f"{'YOLOv8':<20}"
#         f"{'YOLO11':<20}"
#     )

#     print("-" * 75)

#     print(
#         f"{'Images':<30}"
#         f"{v8[0]:<20}"
#         f"{y11[0]:<20}"
#     )

#     print(
#         f"{'Plates detected':<30}"
#         f"{v8[1]:<20}"
#         f"{y11[1]:<20}"
#     )

#     print(
#         f"{'OCR outputs':<30}"
#         f"{v8[2]:<20}"
#         f"{y11[2]:<20}"
#     )

#     print(
#         f"{'Avg time (ms/image)':<30}"
#         f"{v8[3]:<20.2f}"
#         f"{y11[3]:<20.2f}"
#     )

#     print(
#         f"{'Approx FPS':<30}"
#         f"{v8[4]:<20.2f}"
#         f"{y11[4]:<20.2f}"
#     )

#     print("=" * 75)

#     print()
#     print(
#         "CSV:"
#     )

#     print(
#         OUTPUT_FILE
#     )

#     print()
#     print(
#         "NOTE: OCR output count is NOT OCR accuracy."
#     )

#     print(
#         "Ground-truth plate numbers are required "
#         "for true OCR accuracy."
#     )

from ultralytics import YOLO

model8 = YOLO(r"E:\anpr_project\yolo_8.pt")
model11 = YOLO(r"E:\anpr_project\yolo_11.pt")
print("\nEvaluating YOLOv8...")

results8 = model8.val(
    data=r"E:\yolo11_dataset.zip\data.yaml",
    split="test",
    imgsz=640,
    batch=16
)

print("\nEvaluating YOLO11...")

results11 = model11.val(
    data=r"E:\yolo11_dataset.zip\data.yaml",
    split="test",
    imgsz=640,
    batch=16
)