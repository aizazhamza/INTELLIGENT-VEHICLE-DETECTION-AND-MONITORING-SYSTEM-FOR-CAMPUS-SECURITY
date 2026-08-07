# -----------------------------------------------------------
# Require: RecognizedPlates, DatabaseConnection
# Ensure: RecordsStored
# -----------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import cv2


# =========================
# SAVE IMAGE
# =========================
def SavePlateImage(image, save_dir: str = "plate_images") -> str:
    """Save cropped plate image and return path."""

    if image is None:
        return ""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    image_path = os.path.join(save_dir, filename)

    cv2.imwrite(image_path, image)

    return image_path


# =========================
# STORE DATA
# =========================
def StorePlateData(RecognizedPlates: List[Dict], DatabaseConnection):
    """
    Inserts recognized plates into MySQL Events table
    """

    if not RecognizedPlates:
        print("⚠️ No plates to store")
        return 0

    entries_to_insert = []

    for record in RecognizedPlates:

        try:
            plate_number = record.get("plate_number", "").strip().upper()
            timestamp = record.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            location = record.get("location", "")
            event_type = record.get("event_type", "Entry")
            image = record.get("image")

            # Save image first
            image_path = SavePlateImage(image)

            # Append row (MATCH MYSQL ORDER)
            entries_to_insert.append((
                plate_number,
                event_type,
                image_path,
                0.0,      # confidence default
                1         # gate_number default (change if needed)
            ))

        except Exception as e:
            print("❌ RECORD ERROR:", e)

    try:
        cursor = DatabaseConnection.cursor()

        cursor.executemany("""
            INSERT INTO Events
            (license_plate, event_type, picture_path, confidence, gate_number)
            VALUES (%s, %s, %s, %s, %s)
        """, entries_to_insert)

        DatabaseConnection.commit()

        print(f"💾 SUCCESS: Inserted {len(entries_to_insert)} records")

        return len(entries_to_insert)

    except Exception as e:
        print("❌ DATABASE INSERT FAILED:", e)
        return 0