from ultralytics import YOLO

# =========================
# PATHS
# =========================
YOLO8_MODEL = r"E:\anpr_project\yolo_8.pt"
YOLO11_MODEL = r"E:\anpr_project\best.pt"

DATA_YAML = r"E:\testing_dataset.v1i.yolov11\test_data.yaml"

# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model_path, model_name):
    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    model = YOLO(model_path)

    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        verbose=False
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)

    # F1 calculation
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    # Inference time
    inference_time = float(results.speed["inference"])

    # Approximate FPS
    fps = 1000 / inference_time if inference_time > 0 else 0

    # Model information
    params = sum(p.numel() for p in model.model.parameters())
    
    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map5095": map5095,
        "inference": inference_time,
        "fps": fps,
        "params": params
    }


# =========================
# RUN BOTH MODELS
# =========================
yolo8 = evaluate_model(YOLO8_MODEL, "YOLOv8")
yolo11 = evaluate_model(YOLO11_MODEL, "YOLO11")


# =========================
# PRINT COMPARISON
# =========================
print("\n")
print("=" * 95)
print("                 YOLOv8 vs YOLO11 COMPARISON")
print("=" * 95)

print(
    f"{'Metric':<25}"
    f"{'YOLOv8':>18}"
    f"{'YOLO11':>18}"
    f"{'Better':>18}"
)

print("-" * 95)


def better(metric, v8, v11, higher=True):
    if higher:
        return "YOLOv8" if v8 > v11 else "YOLO11"
    else:
        return "YOLOv8" if v8 < v11 else "YOLO11"


print(
    f"{'Precision':<25}"
    f"{yolo8['precision']:.4f}{'':>13}"
    f"{yolo11['precision']:.4f}{'':>13}"
    f"{better('Precision', yolo8['precision'], yolo11['precision']):>18}"
)

print(
    f"{'Recall':<25}"
    f"{yolo8['recall']:.4f}{'':>13}"
    f"{yolo11['recall']:.4f}{'':>13}"
    f"{better('Recall', yolo8['recall'], yolo11['recall']):>18}"
)

print(
    f"{'F1-score':<25}"
    f"{yolo8['f1']:.4f}{'':>13}"
    f"{yolo11['f1']:.4f}{'':>13}"
    f"{better('F1', yolo8['f1'], yolo11['f1']):>18}"
)

print(
    f"{'mAP@0.5':<25}"
    f"{yolo8['map50']:.4f}{'':>13}"
    f"{yolo11['map50']:.4f}{'':>13}"
    f"{better('mAP50', yolo8['map50'], yolo11['map50']):>18}"
)

print(
    f"{'mAP@0.5:0.95':<25}"
    f"{yolo8['map5095']:.4f}{'':>13}"
    f"{yolo11['map5095']:.4f}{'':>13}"
    f"{better('mAP50-95', yolo8['map5095'], yolo11['map5095']):>18}"
)

print(
    f"{'Inference (ms/image)':<25}"
    f"{yolo8['inference']:.2f}{'':>12}"
    f"{yolo11['inference']:.2f}{'':>12}"
    f"{better('Inference', yolo8['inference'], yolo11['inference'], False):>18}"
)

print(
    f"{'Approx. FPS':<25}"
    f"{yolo8['fps']:.2f}{'':>13}"
    f"{yolo11['fps']:.2f}{'':>13}"
    f"{better('FPS', yolo8['fps'], yolo11['fps']):>18}"
)

print(
    f"{'Parameters':<25}"
    f"{yolo8['params'] / 1e6:.2f}M{'':>10}"
    f"{yolo11['params'] / 1e6:.2f}M{'':>10}"
    f"{'—':>18}"
)

print("=" * 95)

print("\nEvaluation completed successfully.")
print("Both models were evaluated on the same test dataset.")