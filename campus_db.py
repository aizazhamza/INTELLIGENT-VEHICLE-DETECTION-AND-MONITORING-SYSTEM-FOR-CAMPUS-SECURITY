import mysql.connector
import traceback
from datetime import datetime


# =========================
# DATABASE CONNECTION
# =========================
def connect_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="hamza221088",
        database="campus"
    )


# =========================
# UTILITIES
# =========================
def normalize_plate(plate):
    return plate.strip().upper().replace(" ", "")


# =========================
# VEHICLES
# =========================
def ensure_vehicle_exists(conn, plate_number):

    plate_number = normalize_plate(plate_number)

    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO Vehicles (license_plate)
            VALUES (%s)
            ON DUPLICATE KEY UPDATE
            license_plate = VALUES(license_plate)
        """, (plate_number,))

        conn.commit()

    finally:
        cursor.close()


# =========================
# EVENTS
# =========================
def get_last_event(conn, plate_number):

    plate_number = normalize_plate(plate_number)

    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT event_type, timestamp
            FROM Events
            WHERE license_plate = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (plate_number,))

        return cursor.fetchone()

    finally:
        cursor.close()


def can_exit(conn, plate_number, min_seconds=30):

    plate_number = normalize_plate(plate_number)

    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT timestamp
            FROM Events
            WHERE license_plate = %s
            AND event_type = 'Entry'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (plate_number,))

        row = cursor.fetchone()

        if row is None:
            return False

        stay_time = (
            datetime.now() - row["timestamp"]
        ).total_seconds()

        print(f"⏱ Stay Time: {stay_time:.1f}s")

        return stay_time >= min_seconds

    finally:
        cursor.close()


def detect_event_type(conn, plate_number):

    last = get_last_event(conn, plate_number)

    if last is None:
        return "Entry"

    if last["event_type"] == "Entry":

        if can_exit(conn, plate_number):
            return "Exit"
        stay_seconds = (
            datetime.now() - last["timestamp"]
        ).total_seconds()

        if stay_seconds < 120:
            return None      # Ignore

        return "Exit"

    return "Entry"

        

  


# =========================
# INSERT EVENT
# =========================
def insert_event(
        conn,
        plate_number,
        event_type,
        image_path,
        confidence,
        gate_number):

    try:

        conn.ping(reconnect=True, attempts=3, delay=1)

        plate_number = normalize_plate(plate_number)

        ensure_vehicle_exists(conn, plate_number)

        cursor = conn.cursor()

        try:

            cursor.execute("""
                INSERT INTO Events
                (
                    license_plate,
                    event_type,
                    picture_path,
                    confidence,
                    gate_number
                )
                VALUES (%s,%s,%s,%s,%s)
            """, (
                plate_number,
                event_type,
                image_path,
                float(confidence),
                int(gate_number)
            ))

            conn.commit()

            print(
                f"💾 DB INSERT SUCCESS: "
                f"{plate_number} {event_type}"
            )

        finally:
            cursor.close()

    except mysql.connector.Error as e:

        print("❌ MYSQL ERROR:", e)
        traceback.print_exc()

    except Exception as e:

        print("❌ GENERAL ERROR:", e)
        traceback.print_exc()
        
        
        
        