import os
import subprocess
import cv2
import json

from flask import (
    Flask, render_template, Response,
    jsonify, request, redirect
)
from werkzeug.utils import secure_filename

import campus_db

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SETTINGS_FILE = "settings.json"




# ================= HOME =================
@app.route('/')
def home():
    conn = campus_db.connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT license_plate, event_type, picture_path, gate_number
        FROM Events
        ORDER BY event_id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    plates = [
        {
            "plate": p,
            "event": e,
            "image": i,
            "gate": g
        }
        for p, e, i, g in rows
    ]

    settings = load_settings()

    return render_template(
    "index.html",
    plates=plates,
    settings=settings
)


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

    conn = campus_db.connect_database()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT license_plate, event_type, picture_path, timestamp, gate_number
        FROM Events
        ORDER BY event_id DESC
        LIMIT 20
    """)
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) AS total FROM Events")
    total_detections = cursor.fetchone()["total"] # type: ignore

    cursor.execute("""
        SELECT COUNT(*) AS today_total
        FROM Events
        WHERE DATE(timestamp) = CURDATE()
    """)
    today_detections = cursor.fetchone()["today_total"] # type: ignore

    cursor.execute("""
        SELECT COUNT(DISTINCT license_plate) AS unique_vehicles
        FROM Events
    """)
    unique_vehicles = cursor.fetchone()["unique_vehicles"] # type: ignore

    cursor.close()
    conn.close()

    plates = [
        {
            "plate": r["license_plate"], # type: ignore
            "event": r["event_type"], # pyright: ignore[reportCallIssue] # type: ignore
            "image": r["picture_path"], # type: ignore
            "time": str(r["timestamp"]), # type: ignore
            "gate": r["gate_number"] # type: ignore
        }
        for r in rows
    ]

    return render_template(
        "dashboard.html",
        plates=plates,
        total_detections=total_detections,
        today_detections=today_detections,
        unique_vehicles=unique_vehicles,
        active_cameras=1
    )


# ================= HISTORY =================
@app.route('/history')
def history():
    conn = campus_db.connect_database()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT license_plate, event_type, picture_path, timestamp, gate_number
        FROM Events
        ORDER BY event_id DESC
        LIMIT 100
    """)

    rows = cursor.fetchall()
    conn.close()

    print("DEBUG HISTORY ROWS:", rows)  # IMPORTANT CHECK

    return render_template("history.html", history=rows)
# ================= SETTINGS =================
import json
import os
from flask import request, redirect, jsonify, render_template

SETTINGS_FILE = "settings.json"


# ================= LOAD SETTINGS =================
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)

    return {
        "confidence": 50,
        "ocr": 50,
        "live_feed": True,
        "save_images": False
    }


# ================= SETTINGS PAGE =================
@app.route('/settings')
def settings():
    data = load_settings()
    return render_template("settings.html", settings=data)


# ================= SAVE SETTINGS =================
@app.route('/save_settings', methods=['POST'])
def save_settings():

    settings = {
        "confidence": int(request.form.get("confidence", 50)),
        "ocr": int(request.form.get("ocr", 50)),

        # FIXED checkbox handling
        "live_feed": request.form.get("live_feed") == "1",
        "save_images": request.form.get("save_images") == "1"
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

    return redirect("/settings")


# ================= API (LIVE SETTINGS) =================
@app.route('/api/settings')
def api_settings():
    return jsonify(load_settings())

# ================= VIDEO UPLOAD =================
@app.route('/upload_video', methods=['POST'])
def upload_video():

    if 'video' not in request.files:
        return "No file uploaded"

    video = request.files['video']
    filename = secure_filename(video.filename or "video.mp4")

    path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(path)

    subprocess.Popen([
        "python", "main.py",
        "--source1", path
    ])

    return "Video uploaded successfully"


# ================= LIVE CAMERA =================
cap = cv2.VideoCapture(0)

def generate_frames():
    global cap

    while True:
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/video')
def video():

    settings = load_settings()

    if not settings["live_feed"]:
        return "Live feed is disabled", 403

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ================= API =================
@app.route('/plates/filename')
def plates():

    conn = campus_db.connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT license_plate, event_type,
               picture_path, timestamp, gate_number
        FROM Events
        ORDER BY event_id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {
            "plate": p,
            "event": e,
            "image": i,
            "time": str(t),
            "gate": g
        }
        for p, e, i, t, g in rows
    ])


# ================= MAIN =================
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)