# -----------------------------------------------------------
# Require: IncomingAPIRequest
# Ensure: APIResponse
# -----------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional


# -----------------------------
# Helper Functions
# -----------------------------

def CreateErrorResponse(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def CreateSuccessResponse(data: Any) -> Dict[str, Any]:
    return {"status": "success", "data": data}


def ValidateRequestFormat(request: Any) -> Optional[str]:
    """Return None if valid else an error message."""
    if not isinstance(request, dict):
        return "Request must be a dictionary"
    if "method" not in request:
        return "Missing required field: method"
    method = request.get("method")
    if method not in {"GET", "POST", "UPDATE", "DELETE"}:
        return f"Unsupported request method: {method}"
    return None


def FetchLicensePlateRecords(parameters: Dict[str, Any], db):
    """Flexible GET.

    Supported parameters:
      - plate_number: exact match
      - hours: only last N hours (uses SQLite datetime)
      - limit: max rows
    """
    plate_number = parameters.get("plate_number")
    hours = parameters.get("hours")
    limit = parameters.get("limit")

    q = "SELECT id, plate_number, timestamp, location, image_path FROM plates"
    clauses = []
    values = []

    if plate_number:
        clauses.append("plate_number = ?")
        values.append(plate_number)

    if hours is not None:
        try:
            hours_int = int(hours)
        except Exception:
            return CreateErrorResponse("hours must be an integer")
        clauses.append("timestamp >= datetime('now', ?)")
        values.append(f"-{hours_int} hours")

    if clauses:
        q += " WHERE " + " AND ".join(clauses)

    q += " ORDER BY timestamp DESC"

    if limit is not None:
        try:
            limit_int = int(limit)
        except Exception:
            return CreateErrorResponse("limit must be an integer")
        q += " LIMIT ?"
        values.append(limit_int)

    cursor = db.cursor()
    cursor.execute(q, tuple(values))
    return cursor.fetchall()


def StoreNewPlateRecord(payload: Dict[str, Any], db):
    required = {"plate_number", "timestamp", "location", "image_path"}
    missing = [k for k in required if k not in payload]
    if missing:
        return CreateErrorResponse(f"Missing payload fields: {', '.join(missing)}")

    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO plates (plate_number, timestamp, location, image_path)
        VALUES (?, ?, ?, ?)
        """,
        (
            payload["plate_number"],
            payload["timestamp"],
            payload.get("location", ""),
            payload.get("image_path", ""),
        ),
    )
    db.commit()
    return {"message": "Record stored successfully"}


def UpdatePlateRecord(payload: Dict[str, Any], db):
    if "plate_number" not in payload:
        return CreateErrorResponse("Missing payload field: plate_number")

    # Allow updating location and/or image_path
    fields = []
    values = []
    if "location" in payload:
        fields.append("location = ?")
        values.append(payload["location"])
    if "image_path" in payload:
        fields.append("image_path = ?")
        values.append(payload["image_path"])

    if not fields:
        return CreateErrorResponse("Nothing to update. Provide location and/or image_path.")

    values.append(payload["plate_number"])

    cursor = db.cursor()
    cursor.execute(
        f"UPDATE plates SET {', '.join(fields)} WHERE plate_number = ?",
        tuple(values),
    )
    db.commit()
    return {"message": "Record updated successfully", "rows_affected": cursor.rowcount}


def DeletePlateRecord(parameters: Dict[str, Any], db):
    if "plate_number" not in parameters:
        return CreateErrorResponse("Missing parameter: plate_number")

    cursor = db.cursor()
    cursor.execute(
        "DELETE FROM plates WHERE plate_number = ?",
        (parameters["plate_number"],),
    )
    db.commit()
    return {"message": "Record deleted successfully", "rows_affected": cursor.rowcount}


# -----------------------------------------------------------
# 1: function ProcessAPIRequest(IncomingAPIRequest)
# -----------------------------------------------------------
def ProcessAPIRequest(IncomingAPIRequest: Dict[str, Any], db):
    err = ValidateRequestFormat(IncomingAPIRequest)
    if err:
        return CreateErrorResponse(err)

    method = IncomingAPIRequest["method"]

    if method == "GET":
        response = FetchLicensePlateRecords(IncomingAPIRequest.get("parameters", {}), db)
        # Fetch can return an error dict
        if isinstance(response, dict) and response.get("status") == "error":
            return response
        return CreateSuccessResponse(response)

    if method == "POST":
        response = StoreNewPlateRecord(IncomingAPIRequest.get("payload", {}), db)
        if isinstance(response, dict) and response.get("status") == "error":
            return response
        return CreateSuccessResponse(response)

    if method == "UPDATE":
        response = UpdatePlateRecord(IncomingAPIRequest.get("payload", {}), db)
        if isinstance(response, dict) and response.get("status") == "error":
            return response
        return CreateSuccessResponse(response)

    if method == "DELETE":
        response = DeletePlateRecord(IncomingAPIRequest.get("parameters", {}), db)
        if isinstance(response, dict) and response.get("status") == "error":
            return response
        return CreateSuccessResponse(response)

    return CreateErrorResponse(f"Unsupported request method: {method}")

# 17: end function
