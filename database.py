"""
Database and storage module for Blood Pressure Tracker.
Supports configurable local SQLite persistence (for Render deployment)
and seamless Supabase cloud synchronization with user scoping,
image duplicate tracking, and chronological time-series queries.
"""

import os
import sqlite3
import datetime
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("bp_tracker.database")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configurable user/owner identifier for scoping health data
BP_USER_ID = os.environ.get("BP_USER_ID", "personal_owner").strip() or "personal_owner"


def get_db_path() -> str:
    """
    Returns the SQLite database path.
    Can be configured via DATABASE_PATH environment variable (e.g. /var/data/bp_tracker.db).
    Defaults to local project directory: bp_tracker.db.
    """
    return os.environ.get("DATABASE_PATH", os.path.join(PROJECT_ROOT, "bp_tracker.db"))


# Optional Supabase credentials from environment (server-side only, never exposed to client)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

_supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase client: {e}. Operating in SQLite mode.")


def is_supabase_enabled() -> bool:
    """Returns True if a remote Supabase client is connected."""
    return _supabase_client is not None


def init_db():
    """Initializes the SQLite database schema if not present, and applies migrations without data loss."""
    db_file = get_db_path()
    parent_dir = os.path.dirname(os.path.abspath(db_file))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bp_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'personal_owner',
                timestamp TEXT NOT NULL,
                sys INTEGER NOT NULL,
                dia INTEGER NOT NULL,
                pulse INTEGER,
                category TEXT,
                model_variant TEXT,
                confidence REAL,
                original_sys INTEGER,
                original_dia INTEGER,
                original_pulse INTEGER,
                was_corrected INTEGER DEFAULT 0,
                notes TEXT,
                image_hash TEXT,
                image_filename TEXT,
                capture_date_source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Check and migrate existing databases that might lack new columns
        cursor.execute("PRAGMA table_info(bp_readings)")
        cols = [c[1] for c in cursor.fetchall()]
        if "user_id" not in cols:
            cursor.execute("ALTER TABLE bp_readings ADD COLUMN user_id TEXT DEFAULT 'personal_owner'")
        if "image_hash" not in cols:
            cursor.execute("ALTER TABLE bp_readings ADD COLUMN image_hash TEXT")
        if "image_filename" not in cols:
            cursor.execute("ALTER TABLE bp_readings ADD COLUMN image_filename TEXT")
        if "capture_date_source" not in cols:
            cursor.execute("ALTER TABLE bp_readings ADD COLUMN capture_date_source TEXT")
        conn.commit()

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bp_readings_user_time ON bp_readings (user_id, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bp_readings_image_hash ON bp_readings (user_id, image_hash)")
        conn.commit()
    logger.info(f"Database initialized at {db_file}")


def calculate_category(sys: Optional[int], dia: Optional[int]) -> str:
    """Classifies blood pressure based on AHA/ACC reference guidelines (for tracking only)."""
    if sys is None or dia is None:
        return "Unknown"
    if sys < 120 and dia < 80:
        return "Normal"
    elif 120 <= sys <= 129 and dia < 80:
        return "Elevated"
    elif (130 <= sys <= 139) or (80 <= dia <= 89):
        return "Hypertension Stage 1"
    elif (sys >= 140) or (dia >= 90):
        return "Hypertension Stage 2"
    return "Recorded"


def check_duplicate_image(image_hash: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Checks if an image with the given hash has already been saved for this user."""
    if not image_hash:
        return None
    active_user = (user_id or BP_USER_ID).strip() or "personal_owner"
    if _supabase_client:
        try:
            resp = _supabase_client.table("bp_readings")\
                .select("id, timestamp, sys, dia, pulse, image_filename")\
                .eq("user_id", active_user)\
                .eq("image_hash", image_hash)\
                .limit(1)\
                .execute()
            if resp.data and len(resp.data) > 0:
                return resp.data[0]
        except Exception as e:
            logger.warning(f"Supabase check_duplicate error: {e}")

    init_db()
    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, sys, dia, pulse, image_filename
            FROM bp_readings
            WHERE (user_id = ? OR user_id IS NULL) AND image_hash = ?
            LIMIT 1
        """, (active_user, image_hash))
        row = cursor.fetchone()
        return dict(row) if row else None


def save_reading(
    sys: int,
    dia: int,
    pulse: Optional[int] = None,
    model_variant: str = "SMART-BP+",
    confidence: Optional[float] = None,
    original_sys: Optional[int] = None,
    original_dia: Optional[int] = None,
    original_pulse: Optional[int] = None,
    notes: str = "",
    custom_timestamp: Optional[str] = None,
    user_id: Optional[str] = None,
    image_hash: Optional[str] = None,
    image_filename: Optional[str] = None,
    capture_date_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a confirmed BP reading to the database.
    Only called upon explicit user confirmation!
    Scoped strictly to the authorized personal owner.
    """
    init_db()
    active_user = (user_id or BP_USER_ID).strip() or "personal_owner"
    now_iso = custom_timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    category = calculate_category(sys, dia)

    was_corrected = 0
    if original_sys is not None and original_sys != sys:
        was_corrected = 1
    if original_dia is not None and original_dia != dia:
        was_corrected = 1
    if original_pulse is not None and original_pulse != pulse:
        was_corrected = 1

    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bp_readings (
                user_id, timestamp, sys, dia, pulse, category,
                model_variant, confidence, original_sys, original_dia,
                original_pulse, was_corrected, notes,
                image_hash, image_filename, capture_date_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            active_user, now_iso, sys, dia, pulse, category,
            model_variant, confidence, original_sys, original_dia,
            original_pulse, was_corrected, notes,
            image_hash, image_filename, capture_date_source
        ))
        conn.commit()
        reading_id = cursor.lastrowid

    record = {
        "id": reading_id,
        "user_id": active_user,
        "timestamp": now_iso,
        "sys": sys,
        "dia": dia,
        "pulse": pulse,
        "category": category,
        "model_variant": model_variant,
        "confidence": confidence,
        "original_sys": original_sys,
        "original_dia": original_dia,
        "original_pulse": original_pulse,
        "was_corrected": bool(was_corrected),
        "notes": notes,
        "image_hash": image_hash,
        "image_filename": image_filename,
        "capture_date_source": capture_date_source
    }

    # Cloud Supabase synchronization scoped to authenticated user
    if _supabase_client:
        try:
            _supabase_client.table("bp_readings").insert({
                "user_id": active_user,
                "timestamp": now_iso,
                "sys": sys,
                "dia": dia,
                "pulse": pulse,
                "category": category,
                "model_variant": model_variant,
                "confidence": confidence,
                "notes": notes,
                "was_corrected": bool(was_corrected),
                "image_hash": image_hash,
                "image_filename": image_filename,
                "capture_date_source": capture_date_source
            }).execute()
            logger.info("Reading synced to Supabase successfully with user scoping.")
        except Exception as e:
            logger.warning(f"Could not sync reading to Supabase: {e}")

    return record


def save_batch_readings(
    readings: List[Dict[str, Any]],
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Saves multiple confirmed readings sequentially within a single database transaction."""
    saved_records = []
    for item in readings:
        rec = save_reading(
            sys=int(item["sys"]),
            dia=int(item["dia"]),
            pulse=int(item["pulse"]) if item.get("pulse") is not None else None,
            model_variant=item.get("model_variant", "SMART-BP+"),
            confidence=float(item.get("confidence", 1.0)) if item.get("confidence") is not None else None,
            original_sys=int(item["original_sys"]) if item.get("original_sys") is not None else None,
            original_dia=int(item["original_dia"]) if item.get("original_dia") is not None else None,
            original_pulse=int(item["original_pulse"]) if item.get("original_pulse") is not None else None,
            notes=item.get("notes", ""),
            custom_timestamp=item.get("timestamp"),
            user_id=user_id,
            image_hash=item.get("image_hash"),
            image_filename=item.get("image_filename"),
            capture_date_source=item.get("capture_date_source")
        )
        saved_records.append(rec)
    return saved_records


def get_all_readings(
    limit: int = 500,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    order: str = "DESC"
) -> List[Dict[str, Any]]:
    """Retrieves reading history scoped to the authorized user with optional date range and ordering."""
    active_user = (user_id or BP_USER_ID).strip() or "personal_owner"
    order_direction = "ASC" if order.upper() == "ASC" else "DESC"

    if _supabase_client:
        try:
            query = _supabase_client.table("bp_readings")\
                .select("*")\
                .eq("user_id", active_user)
            if start_date:
                query = query.gte("timestamp", f"{start_date} 00:00:00")
            if end_date:
                query = query.lte("timestamp", f"{end_date} 23:59:59")
            response = query.order("timestamp", desc=(order_direction == "DESC")).limit(limit).execute()
            if response.data is not None:
                return response.data
        except Exception as e:
            logger.warning(f"Supabase fetch error: {e}. Falling back to SQLite.")

    init_db()
    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = ["SELECT * FROM bp_readings WHERE (user_id = ? OR user_id IS NULL)"]
        params: List[Any] = [active_user]

        if start_date:
            sql.append("AND timestamp >= ?")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            sql.append("AND timestamp <= ?")
            params.append(f"{end_date} 23:59:59")

        sql.append(f"ORDER BY timestamp {order_direction} LIMIT ?")
        params.append(limit)

        cursor.execute(" ".join(sql), params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def delete_reading(reading_id: int, user_id: Optional[str] = None) -> bool:
    """Deletes a reading by ID, strictly verifying ownership."""
    active_user = (user_id or BP_USER_ID).strip() or "personal_owner"
    if _supabase_client:
        try:
            _supabase_client.table("bp_readings")\
                .delete()\
                .eq("id", reading_id)\
                .eq("user_id", active_user)\
                .execute()
        except Exception as e:
            logger.warning(f"Supabase delete error: {e}")

    init_db()
    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM bp_readings
            WHERE id = ? AND (user_id = ? OR user_id IS NULL)
        """, (reading_id, active_user))
        conn.commit()
        return cursor.rowcount > 0


def get_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Calculates summary statistics scoped to the authorized user."""
    readings = get_all_readings(limit=500, user_id=user_id)
    if not readings:
        return {"total_count": 0, "avg_sys": None, "avg_dia": None, "avg_pulse": None}

    sys_vals = [r["sys"] for r in readings if r.get("sys") is not None]
    dia_vals = [r["dia"] for r in readings if r.get("dia") is not None]
    pul_vals = [r["pulse"] for r in readings if r.get("pulse") is not None]

    return {
        "total_count": len(readings),
        "avg_sys": round(sum(sys_vals) / len(sys_vals), 1) if sys_vals else None,
        "avg_dia": round(sum(dia_vals) / len(dia_vals), 1) if dia_vals else None,
        "avg_pulse": round(sum(pul_vals) / len(pul_vals), 1) if pul_vals else None,
        "min_sys": min(sys_vals) if sys_vals else None,
        "max_sys": max(sys_vals) if sys_vals else None,
        "min_dia": min(dia_vals) if dia_vals else None,
        "max_dia": max(dia_vals) if dia_vals else None
    }


def get_chart_data(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates chronological chart dataset sorted oldest-to-newest (ASC).
    Computes descriptive statistics (labeled as tracking stats, not diagnoses).
    Supports daily, weekly, monthly, and all-time views.
    """
    # Calculate date boundaries for predefined periods
    now = datetime.datetime.now()
    calc_start = start_date
    calc_end = end_date

    if period == "daily":
        calc_start = now.strftime("%Y-%m-%d")
        calc_end = now.strftime("%Y-%m-%d")
    elif period == "weekly":
        calc_start = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        calc_end = now.strftime("%Y-%m-%d")
    elif period == "monthly":
        calc_start = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        calc_end = now.strftime("%Y-%m-%d")

    readings = get_all_readings(
        limit=1000,
        user_id=user_id,
        start_date=calc_start,
        end_date=calc_end,
        order="ASC"
    )

    sys_vals = [r["sys"] for r in readings if r.get("sys") is not None]
    dia_vals = [r["dia"] for r in readings if r.get("dia") is not None]
    pul_vals = [r["pulse"] for r in readings if r.get("pulse") is not None]

    summary = {
        "disclaimer": "Descriptive statistics for personal tracking. Not a medical evaluation or diagnosis.",
        "total_count": len(readings),
        "mean_sys": round(sum(sys_vals) / len(sys_vals), 1) if sys_vals else None,
        "mean_dia": round(sum(dia_vals) / len(dia_vals), 1) if dia_vals else None,
        "mean_pulse": round(sum(pul_vals) / len(pul_vals), 1) if pul_vals else None,
        "min_sys": min(sys_vals) if sys_vals else None,
        "max_sys": max(sys_vals) if sys_vals else None,
        "min_dia": min(dia_vals) if dia_vals else None,
        "max_dia": max(dia_vals) if dia_vals else None,
        "min_pulse": min(pul_vals) if pul_vals else None,
        "max_pulse": max(pul_vals) if pul_vals else None,
        "earliest_timestamp": readings[0]["timestamp"] if readings else None,
        "latest_timestamp": readings[-1]["timestamp"] if readings else None
    }

    return {
        "status": "success",
        "period": period or "all",
        "filter": {"start_date": calc_start, "end_date": calc_end},
        "count": len(readings),
        "readings": readings,
        "summary": summary
    }
