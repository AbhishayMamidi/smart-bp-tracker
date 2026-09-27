"""
Local Storage and File Persistence Module for SMART-BP Blood Pressure Tracker.
Manages local image storage in data/uploads/, safe path resolution without directory
traversal vulnerabilities, file deduplication via SHA-256 hash, and database backups.
"""

import os
import shutil
import zipfile
import datetime
import logging
from typing import Optional, Tuple

logger = logging.getLogger("bp_tracker.storage")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_uploads_dir() -> str:
    """
    Returns the absolute path to the uploads directory.
    Configurable via UPLOADS_DIR environment variable.
    Defaults to data/uploads/ under PROJECT_ROOT.
    """
    d = os.environ.get("UPLOADS_DIR") or os.path.join(PROJECT_ROOT, "data", "uploads")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def get_backups_dir() -> str:
    """Returns the absolute path to the backup storage directory."""
    d = os.environ.get("BACKUPS_DIR") or os.path.join(PROJECT_ROOT, "data", "backups")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def save_image_bytes(image_bytes: bytes, image_hash: str, original_filename: str = "") -> str:
    """
    Saves raw image bytes to the local uploads directory using its SHA-256 hash.
    Returns the relative path string (e.g. 'uploads/{hash}.jpg') suitable for DB storage.
    Guarantees that files cannot overwrite outside get_uploads_dir().
    """
    ext = ".jpg"
    if original_filename:
        _, file_ext = os.path.splitext(original_filename)
        if file_ext.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = file_ext.lower()

    safe_filename = f"{image_hash}{ext}"
    uploads_dir = get_uploads_dir()
    filepath = os.path.join(uploads_dir, safe_filename)

    if not os.path.exists(filepath):
        try:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Saved original photograph: {safe_filename} ({len(image_bytes)} bytes)")
        except Exception as e:
            logger.error(f"Failed to save image file {safe_filename}: {e}")

    return f"uploads/{safe_filename}"


def get_image_file_path(filename_or_path: str) -> Optional[str]:
    """
    Resolves an image filename or relative path to a verified safe absolute path.
    Prevents arbitrary filesystem traversal attacks using os.path.basename and realpath check.
    Returns None if file does not exist or falls outside the uploads directory.
    """
    if not filename_or_path:
        return None

    # Strip any leading 'uploads/' or directory separators to prevent traversal
    clean_name = os.path.basename(filename_or_path.strip().replace("\\", "/"))
    uploads_dir = get_uploads_dir()
    resolved_path = os.path.abspath(os.path.join(uploads_dir, clean_name))

    # Strict security check: must reside inside uploads_dir
    common_prefix = os.path.commonpath([uploads_dir, resolved_path])
    if common_prefix != uploads_dir:
        logger.warning(f"Directory traversal attempt blocked: {filename_or_path}")
        return None

    if os.path.isfile(resolved_path):
        return resolved_path

    return None


def delete_image_file(filename_or_path: str) -> bool:
    """
    Safely deletes a stored image file from the uploads directory.
    Returns True if deleted, False otherwise.
    """
    file_path = get_image_file_path(filename_or_path)
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted image file: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            logger.error(f"Could not delete image file {file_path}: {e}")
            return False
    return False


def create_backup(dest_zip_path: Optional[str] = None) -> str:
    """
    Creates a zip archive containing the SQLite database and all uploaded photographs.
    Returns the path to the created zip file.
    """
    import database
    db_path = database.get_db_path()
    uploads_dir = get_uploads_dir()
    backups_dir = get_backups_dir()

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not dest_zip_path:
        dest_zip_path = os.path.join(backups_dir, f"bp_backup_{timestamp_str}.zip")

    with zipfile.ZipFile(dest_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(db_path):
            zf.write(db_path, arcname="bp_tracker.db")

        if os.path.exists(uploads_dir):
            for root, _, files in os.walk(uploads_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, uploads_dir)
                    zf.write(full_p, arcname=os.path.join("uploads", rel_p))

    logger.info(f"Database and uploads backup created at {dest_zip_path}")
    return dest_zip_path


def restore_backup(backup_zip_path: str) -> Tuple[bool, str]:
    """
    Restores SQLite database and uploads directory from a backup zip archive.
    Returns (success: bool, message: str).
    """
    if not os.path.exists(backup_zip_path):
        return False, f"Backup file not found: {backup_zip_path}"

    import database
    db_path = database.get_db_path()
    uploads_dir = get_uploads_dir()

    try:
        with zipfile.ZipFile(backup_zip_path, "r") as zf:
            for member in zf.namelist():
                # Prevent Zip Slip directory traversal vulnerability
                target_p = os.path.abspath(os.path.join(PROJECT_ROOT, "data", member))
                if not target_p.startswith(os.path.abspath(os.path.join(PROJECT_ROOT, "data"))):
                    return False, f"Illegal entry in zip archive: {member}"

                if member == "bp_tracker.db":
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    with zf.open(member) as src, open(db_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                elif member.startswith("uploads/"):
                    sub_file = member[len("uploads/"):]
                    if sub_file:
                        dest_f = os.path.join(uploads_dir, os.path.basename(sub_file))
                        with zf.open(member) as src, open(dest_f, "wb") as dst:
                            shutil.copyfileobj(src, dst)

        return True, "Backup restored successfully."
    except Exception as e:
        logger.exception("Restore failed")
        return False, f"Restore failed: {e}"
