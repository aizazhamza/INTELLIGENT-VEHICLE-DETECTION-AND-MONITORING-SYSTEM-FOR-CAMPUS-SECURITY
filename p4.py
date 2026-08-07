# p4.py - STABLE TRACKER FIXED

from __future__ import annotations
import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except:
    linear_sum_assignment = None


# =========================
# TRACK ID GENERATOR
# =========================
class TrackIDGenerator:
    current_id = 0

    @classmethod
    def next_id(cls):
        cls.current_id += 1
        return cls.current_id


# =========================
# IOU
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


# =========================
# TRACK CLASS
# =========================
class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.bbox = bbox
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0


# =========================
# MATCHING
# =========================
def match(tracks, detections, threshold=0.3):
    matches = []
    used = set()

    for ti, t in enumerate(tracks):
        best_iou = 0
        best_di = -1

        for di, d in enumerate(detections):
            if di in used:
                continue

            iou = compute_iou(t.bbox, d)
            if iou > best_iou:
                best_iou = iou
                best_di = di

        if best_iou >= threshold:
            matches.append((ti, best_di))
            used.add(best_di)

    return matches, used


# =========================
# MAIN TRACKER (FIXED)
# =========================
def TrackLicensePlates(
    detections,
    tracks,
    association_threshold=0.3,
    max_age=60
):

    # predict
    for t in tracks:
        t.predict()

    matches, used_dets = match(tracks, detections, association_threshold)

    # update matched
    for ti, di in matches:
        tracks[ti].update(detections[di])

    # new tracks
    for i, det in enumerate(detections):
        if i not in used_dets:
            tracks.append(Track(det, TrackIDGenerator.next_id()))

    # remove dead tracks
    tracks = [t for t in tracks if t.time_since_update <= max_age]

    return tracks