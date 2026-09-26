"""
EXIF Metadata Extraction and Image Utility Module for SMART-BP Blood Pressure Tracker.
Extracts photo capture dates (DateTimeOriginal, DateTimeDigitized, DateTime),
manages Asia/Kolkata timezone normalization, computes SHA-256 duplicate hashes,
and generates lightweight base64 preview thumbnails.
"""

import io
import re
import base64
import hashlib
import logging
import datetime
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ExifTags

logger = logging.getLogger("bp_tracker.exif")

DEFAULT_TIMEZONE = "Asia/Kolkata"

# Standard EXIF tag codes
TAG_DATETIME_ORIGINAL = 36867   # 0x9003
TAG_DATETIME_DIGITIZED = 36868  # 0x9004
TAG_DATETIME = 306              # 0x0132
TAG_SUBSEC_TIME_ORIG = 37521    # 0x9291
TAG_OFFSET_TIME_ORIG = 36881    # 0x9011
TAG_OFFSET_TIME = 36880         # 0x9010
TAG_OFFSET_TIME_DIG = 36882     # 0x9012


def calculate_image_hash(image_bytes: bytes) -> str:
    """Computes SHA-256 hash of the raw image bytes for duplicate detection."""
    return hashlib.sha256(image_bytes).hexdigest()


def parse_exif_date_string(date_str: str) -> Optional[datetime.datetime]:
    """
    Parses an EXIF date string into a datetime object.
    Standard EXIF format: 'YYYY:MM:DD HH:MM:SS'.
    Also supports 'YYYY-MM-DD HH:MM:SS' and ISO formats.
    Returns None if date cannot be parsed or values are nonsensical.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()
    # Reject placeholders like '0000:00:00 00:00:00'
    if cleaned.startswith("0000") or cleaned.startswith("1900:00:00"):
        return None

    # Handle standard EXIF 'YYYY:MM:DD HH:MM:SS'
    patterns = [
        ("%Y:%m:%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y:%m:%d %H:%M", 16),
        ("%Y-%m-%d %H:%M", 16),
        ("%Y-%m-%dT%H:%M:%S", 19),
    ]

    for fmt, length in patterns:
        try:
            return datetime.datetime.strptime(cleaned[:length], fmt)
        except (ValueError, IndexError):
            continue

    # Regex fallback for non-standard separators
    match = re.match(r"^(\d{4})[:\-/](\d{2})[:\-/](\d{2})[ T](\d{2})[:\-](\d{2})[:\-](\d{2})", cleaned)
    if match:
        try:
            y, m, d, H, M, S = [int(v) for v in match.groups()]
            return datetime.datetime(y, m, d, H, M, S)
        except ValueError:
            pass

    return None


def extract_exif_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts original photo capture timestamp and metadata from raw image bytes.
    Preserves original metadata before any OpenCV preprocessing.

    Priority order:
    1. DateTimeOriginal (EXIF 36867)
    2. DateTimeDigitized (EXIF 36868)
    3. DateTime (EXIF 306)
    4. If none found, marks date as unavailable (never fakes or invents a date).
    """
    img_hash = calculate_image_hash(image_bytes)

    result: Dict[str, Any] = {
        "has_capture_date": False,
        "capture_date": None,         # Formatted as "YYYY-MM-DD HH:MM:SS"
        "capture_date_source": None,  # "DateTimeOriginal", "DateTimeDigitized", "DateTime", or None
        "raw_date_string": None,
        "offset": None,               # e.g. "+05:30"
        "subsec": None,
        "timezone": DEFAULT_TIMEZONE,
        "display_datetime": "Capture date unavailable",
        "image_hash": img_hash,
        "image_width": None,
        "image_height": None,
        "image_format": None
    }

    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            result["image_width"], result["image_height"] = pil_img.size
            result["image_format"] = pil_img.format or "JPEG"

            # Extract EXIF dictionary
            exif_dict = {}
            if hasattr(pil_img, "getexif"):
                exif = pil_img.getexif()
                if exif:
                    exif_dict.update(dict(exif))
                    # Check Exif IFD sub-dictionary
                    if hasattr(ExifTags, "IFD") and hasattr(ExifTags.IFD, "Exif"):
                        try:
                            ifd_exif = exif.get_ifd(ExifTags.IFD.Exif)
                            if ifd_exif:
                                exif_dict.update(dict(ifd_exif))
                        except Exception:
                            pass

            # Legacy fallback
            if hasattr(pil_img, "_getexif"):
                try:
                    legacy_exif = pil_img._getexif()
                    if legacy_exif:
                        for k, v in legacy_exif.items():
                            if k not in exif_dict:
                                exif_dict[k] = v
                except Exception:
                    pass

            if not exif_dict:
                return result

            # Extract timezone offset if present
            offset = (
                exif_dict.get(TAG_OFFSET_TIME_ORIG)
                or exif_dict.get(TAG_OFFSET_TIME)
                or exif_dict.get(TAG_OFFSET_TIME_DIG)
            )
            if offset and isinstance(offset, str) and len(offset.strip()) in (5, 6):
                result["offset"] = offset.strip()

            subsec = exif_dict.get(TAG_SUBSEC_TIME_ORIG)
            if subsec:
                result["subsec"] = str(subsec).strip()

            # Priority 1: DateTimeOriginal
            dto_val = exif_dict.get(TAG_DATETIME_ORIGINAL)
            parsed_dt = parse_exif_date_string(str(dto_val) if dto_val else "")
            if parsed_dt:
                result["has_capture_date"] = True
                result["capture_date"] = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                result["capture_date_source"] = "DateTimeOriginal"
                result["raw_date_string"] = str(dto_val).strip()
            else:
                # Priority 2: DateTimeDigitized
                dtd_val = exif_dict.get(TAG_DATETIME_DIGITIZED)
                parsed_dt = parse_exif_date_string(str(dtd_val) if dtd_val else "")
                if parsed_dt:
                    result["has_capture_date"] = True
                    result["capture_date"] = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                    result["capture_date_source"] = "DateTimeDigitized"
                    result["raw_date_string"] = str(dtd_val).strip()
                else:
                    # Priority 3: DateTime
                    dt_val = exif_dict.get(TAG_DATETIME)
                    parsed_dt = parse_exif_date_string(str(dt_val) if dt_val else "")
                    if parsed_dt:
                        result["has_capture_date"] = True
                        result["capture_date"] = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                        result["capture_date_source"] = "DateTime"
                        result["raw_date_string"] = str(dt_val).strip()

            if result["has_capture_date"]:
                offset_str = f" ({result['offset']})" if result["offset"] else f" ({DEFAULT_TIMEZONE})"
                result["display_datetime"] = f"{result['capture_date']}{offset_str}"
            else:
                result["display_datetime"] = "Capture date unavailable"

    except Exception as e:
        logger.warning(f"Failed to read image EXIF metadata: {e}")

    return result


def create_thumbnail_base64(image_bytes: bytes, max_size: Tuple[int, int] = (160, 160)) -> Optional[str]:
    """
    Creates a compact base64 JPEG thumbnail (~1-2KB) for web UI preview.
    Guarantees low memory footprint.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            # Convert RGBA / P mode to RGB for clean JPEG encoding
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail(max_size, Image.Resampling.LANCZOS)
            out_buf = io.BytesIO()
            im.save(out_buf, format="JPEG", quality=75, optimize=True)
            return base64.b64encode(out_buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Could not generate thumbnail: {e}")
        return None
