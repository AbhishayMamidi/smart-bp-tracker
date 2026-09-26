"""
Database and storage module for Blood Pressure Tracker.
Supports configurable local SQLite persistence (for Render Persistent Disks)
and seamless Supabase cloud synchronization.
"""

import os
import sqlite3
import datetime
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("bp_tracker.database")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_db_path() -> str:
    """
    Returns the SQLite database path.
    Can be configured via DATABASE_PATH environment variable (e.g. for Render Persistent Disks: /var/data/bp_tracker.db).
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


def init_db():
    """Initializes the SQLite database schema if not present."""
    db_file = get_db_path()
    parent_dir = os.path.dirname(os.path.abspath(db_file))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bp_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info(f"Database initialized at {db_file}")


def calculate_category(sys: Optional[int], dia: Optional[int]) -> str:
    """Classifies blood pressure based on AHA/ACC guidelines (for tracking only)."""
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
    custom_timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a confirmed BP reading to the database.
    Only called upon explicit user confirmation!
    """
    init_db()
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
                timestamp, sys, dia, pulse, category,
                model_variant, confidence, original_sys, original_dia,
                original_pulse, was_corrected, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso, sys, dia, pulse, category,
            model_variant, confidence, original_sys, original_dia,
            original_pulse, was_corrected, notes
        ))
        conn.commit()
        reading_id = cursor.lastrowid

    record = {
        "id": reading_id,
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
        "notes": notes
    }

    # Cloud Supabase synchronization if configured
    if _supabase_client:
        try:
            _supabase_client.table("bp_readings").insert({
                "timestamp": now_iso,
                "sys": sys,
                "dia": dia,
                "pulse": pulse,
                "category": category,
                "model_variant": model_variant,
                "confidence": confidence,
                "notes": notes,
                "was_corrected": bool(was_corrected)
            }).execute()
            logger.info("Reading synced to Supabase successfully.")
        except Exception as e:
            logger.warning(f"Could not sync reading to Supabase: {e}")

    return record


def get_all_readings(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves reading history sorted newest first."""
    if _supabase_client:
        try:
            response = _supabase_client.table("bp_readings").select("*").order("timestamp", desc=True).limit(limit).execute()
            if response.data is not None:
                return response.data
        except Exception as e:
            logger.warning(f"Supabase fetch error: {e}. Falling back to SQLite.")

    init_db()
    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM bp_readings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def delete_reading(reading_id: int) -> bool:
    """Deletes a reading by ID."""
    if _supabase_client:
        try:
            _supabase_client.table("bp_readings").delete().eq("id", reading_id).execute()
        except Exception as e:
            logger.warning(f"Supabase delete error: {e}")

    init_db()
    db_file = get_db_path()
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bp_readings WHERE id = ?", (reading_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_stats() -> Dict[str, Any]:
    """Calculates summary statistics."""
    readings = get_all_readings(limit=500)
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
