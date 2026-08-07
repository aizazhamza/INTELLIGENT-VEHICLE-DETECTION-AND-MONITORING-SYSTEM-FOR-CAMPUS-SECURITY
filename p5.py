# -----------------------------------------------------------
# Require: DetectionResults, TrackingResults
# Ensure: UpdatedParameters
# -----------------------------------------------------------

from __future__ import annotations

import numpy as np


# Helper: Calculate Mean Confidence
def CalculateMeanConfidence(DetectionResults):
    if not DetectionResults:
        return 0.0
    confidences = [float(det.get("confidence", 0.0)) for det in DetectionResults]
    return float(np.mean(confidences)) if confidences else 0.0


def _track_is_active(t) -> bool:
    """Support both dict-tracks and Track objects."""
    if isinstance(t, dict):
        return bool(t.get("active", False))
    # Track object from p4.py
    return getattr(t, "time_since_update", 9999) == 0


# Helper: Calculate Tracking Stability
def CalculateTrackingStability(TrackingResults):
    if not TrackingResults:
        return 0.0
    active_tracks = [t for t in TrackingResults if _track_is_active(t)]
    return float(len(active_tracks) / len(TrackingResults))


# 1: function AdjustParameters(DetectionResults, TrackingResults)
def AdjustParameters(
    DetectionResults,
    TrackingResults,
    TargetConfidence: float = 0.75,
    TargetStability: float = 0.70,
    detection_threshold: float = 0.5,
    association_threshold: float = 0.3,
):

    UpdatedParameters = {
        "detection_threshold": float(detection_threshold),
        "association_threshold": float(association_threshold),
    }

    # 2: DetectionConfidence ← CalculateMeanConfidence(DetectionResults)
    DetectionConfidence = CalculateMeanConfidence(DetectionResults)

    # 3: if DetectionConfidence < TargetConfidence then
    if DetectionConfidence < TargetConfidence:
        # Loosen detection threshold to let more candidates through
        UpdatedParameters["detection_threshold"] -= 0.05
    else:
        # Tighten if confidence is already strong
        UpdatedParameters["detection_threshold"] += 0.05

    UpdatedParameters["detection_threshold"] = max(0.1, min(0.95, UpdatedParameters["detection_threshold"]))

    # 8: TrackingConsistency ← CalculateTrackingStability(TrackingResults)
    TrackingConsistency = CalculateTrackingStability(TrackingResults)

    # If tracking is unstable, loosen association threshold (allow matches with lower IoU).
    if TrackingConsistency < TargetStability:
        UpdatedParameters["association_threshold"] -= 0.05
    else:
        UpdatedParameters["association_threshold"] += 0.02

    UpdatedParameters["association_threshold"] = max(0.1, min(0.95, UpdatedParameters["association_threshold"]))

    return UpdatedParameters

# 13: end function
