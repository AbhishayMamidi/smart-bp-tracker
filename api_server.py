"""
FastAPI Server for Blood Pressure Tracker & SMART-BP Inference Engine.
Production-hardened configuration for Render Docker deployment and local development.
Supports single-image and sequential batch digitization with EXIF metadata extraction,
SHA-256 duplicate detection, time-series chart data API, and personal access authentication.
"""

import os
import gc
import sys
import base64
import logging
import secrets
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Header, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smart_bp_inference import get_inference_engine, DEFAULT_WEIGHTS
import database
import exif_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bp_api")

# ============================================================================
# Security: Access Token & Single-User Access Gate
# ============================================================================

_env_token = os.environ.get("APP_ACCESS_TOKEN", "").strip()
if _env_token:
    ACTIVE_ACCESS_TOKEN = _env_token
    IS_AUTO_GENERATED_TOKEN = False
else:
    ACTIVE_ACCESS_TOKEN = secrets.token_urlsafe(24)
    IS_AUTO_GENERATED_TOKEN = True


def get_expected_token() -> str:
    """Fetches the current expected access token (supports runtime env overrides in testing)."""
    override = os.environ.get("APP_ACCESS_TOKEN", "").strip()
    return override if override else ACTIVE_ACCESS_TOKEN


def verify_access(
    x_access_token: Optional[str] = Header(None, alias="X-Access-Token"),
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> str:
    """
    Enforces authentication on all personal health routes and inference endpoints.
    Accepts tokens via X-Access-Token header or Authorization: Bearer <token>.
    Uses constant-time secrets.compare_digest to prevent timing attacks.
    """
    token = None
    if x_access_token:
        token = x_access_token.strip()
    elif authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        elif len(parts) == 1:
            token = parts[0].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Access token required for personal health records.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    expected = get_expected_token()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid access token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return token


# ============================================================================
# FastAPI Application & Security Middleware
# ============================================================================

app = FastAPI(
    title="Blood Pressure Tracker & SMART-BP Digitization API",
    description="Automated digital blood pressure monitor digitization using YOLOv8 SMART-BP.",
    version="1.4.0"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Configurable CORS for production deployment
cors_env = os.environ.get("CORS_ORIGINS", "").strip()
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:10000",
        "http://127.0.0.1:10000",
        "http://localhost:3000",
        "http://localhost:5173",
    ]

has_wildcard = "*" in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not has_wildcard,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Upload limits tailored for Render Free Tier (512 MB RAM)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 20)) * 1024 * 1024
MAX_BATCH_IMAGES = int(os.environ.get("MAX_BATCH_IMAGES", 25))
MAX_BATCH_SIZE_BYTES = int(os.environ.get("MAX_BATCH_SIZE_MB", 100)) * 1024 * 1024


# ============================================================================
# Request / Response Schemas
# ============================================================================

class ConfirmReadingRequest(BaseModel):
    sys: int = Field(..., ge=40, le=260, description="Systolic blood pressure (mmHg)")
    dia: int = Field(..., ge=30, le=200, description="Diastolic blood pressure (mmHg)")
    pulse: Optional[int] = Field(None, ge=25, le=240, description="Pulse rate (BPM)")
    model_variant: str = "SMART-BP+"
    confidence: Optional[float] = None
    original_sys: Optional[int] = None
    original_dia: Optional[int] = None
    original_pulse: Optional[int] = None
    notes: Optional[str] = ""
    timestamp: Optional[str] = None
    image_hash: Optional[str] = None
    image_filename: Optional[str] = None
    capture_date_source: Optional[str] = None


class BatchConfirmItem(BaseModel):
    sys: int = Field(..., ge=40, le=260)
    dia: int = Field(..., ge=30, le=200)
    pulse: Optional[int] = Field(None, ge=25, le=240)
    timestamp: Optional[str] = Field(None, description="Capture date/time in YYYY-MM-DD HH:MM:SS format. Required before saving.")
    model_variant: str = "SMART-BP+"
    confidence: Optional[float] = None
    original_sys: Optional[int] = None
    original_dia: Optional[int] = None
    original_pulse: Optional[int] = None
    notes: Optional[str] = ""
    image_hash: Optional[str] = None
    image_filename: Optional[str] = None
    capture_date_source: Optional[str] = None


class BatchConfirmRequest(BaseModel):
    items: List[BatchConfirmItem]


class Base64ImageRequest(BaseModel):
    image_base64: str
    variant: Optional[str] = "SMART-BP+"
    conf_threshold: Optional[float] = 0.2


class VerifyTokenRequest(BaseModel):
    token: str


# ============================================================================
# Health Check Endpoints (Public status probes, sanitized for security)
# ============================================================================

@app.get("/api/health")
@app.get("/health")
@app.get("/healthz")
def health_check():
    """
    Returns engine health, CPU device status, and model availability.
    Sanitized: does not leak internal filesystem paths or secret keys.
    Compatible with Docker HEALTHCHECK and cloud orchestrators.
    """
    weights_status = {
        name: {
            "loaded": os.path.exists(path),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else 0
        }
        for name, path in DEFAULT_WEIGHTS.items()
    }
    all_exist = all(w["loaded"] for w in weights_status.values())

    return {
        "status": "healthy" if all_exist else "degraded",
        "device": "CPU",
        "models": weights_status,
        "default_variant": "SMART-BP+",
        "database": {
            "type": "Supabase (cloud)" if database.is_supabase_enabled() else "SQLite (persistent)",
            "connected": True
        },
        "auth_enabled": True
    }


# ============================================================================
# Auth Management Endpoints
# ============================================================================

@app.post("/api/auth/verify")
def verify_token_endpoint(req: VerifyTokenRequest):
    """Verifies a user-supplied token against the server's active passkey."""
    expected = get_expected_token()
    if secrets.compare_digest(req.token.strip(), expected):
        return {"status": "success", "authenticated": True, "message": "Access granted."}
    raise HTTPException(status_code=401, detail="Invalid access token.")


@app.get("/api/auth/status")
def auth_status_endpoint(
    x_access_token: Optional[str] = Header(None, alias="X-Access-Token"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Checks whether the client's current session or credentials are valid."""
    token = None
    if x_access_token:
        token = x_access_token.strip()
    elif authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        elif len(parts) == 1:
            token = parts[0].strip()

    expected = get_expected_token()
    is_auth = bool(token and secrets.compare_digest(token, expected))
    return {
        "auth_required": True,
        "authenticated": is_auth
    }


# ============================================================================
# Single-Image Inference Endpoints
# ============================================================================

@app.post("/api/read-bp-image")
async def read_bp_image_endpoint(
    image: Optional[UploadFile] = File(None),
    variant: str = Form("SMART-BP+"),
    conf_threshold: float = Form(0.2),
    _token: str = Depends(verify_access)
):
    """
    Single-image inference endpoint. Upload a BP monitor photograph.
    Runs EXIF extraction, checks duplicate status, and executes SMART-BP inference.
    Readings are NOT automatically saved; user confirmation is required!
    """
    if image is None:
        raise HTTPException(status_code=400, detail="No image file provided.")

    if image.content_type and not (image.content_type.startswith("image/") or image.content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image.content_type}'. Must be an image.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file received.")

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum upload limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.")

        # Extract metadata prior to inference
        metadata = exif_utils.extract_exif_metadata(image_bytes)
        duplicate_record = database.check_duplicate_image(metadata["image_hash"])

        engine = get_inference_engine(variant=variant)
        result = engine.predict(
            image_input=image_bytes,
            conf_threshold=conf_threshold,
            generate_annotated_image=True
        )

        sys_val = result["readings"].get("sys")
        dia_val = result["readings"].get("dia")
        result["category"] = database.calculate_category(sys_val, dia_val)
        result["metadata"] = metadata
        result["is_duplicate"] = duplicate_record is not None
        result["duplicate_info"] = duplicate_record
        result["filename"] = image.filename or "monitor_photo.jpg"

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing blood pressure image")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Inference processing failed: {str(e)}",
                "readings": {"sys": None, "dia": None, "pul": None}
            }
        )


@app.post("/api/read-bp-image-base64")
def read_bp_image_base64_endpoint(
    req: Base64ImageRequest,
    _token: str = Depends(verify_access)
):
    """Inference endpoint accepting base64 encoded data URI (for live mobile camera frames)."""
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"Image exceeds limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.")

        engine = get_inference_engine(variant=req.variant or "SMART-BP+")
        result = engine.predict(
            image_input=image_bytes,
            conf_threshold=req.conf_threshold or 0.2,
            generate_annotated_image=True
        )
        sys_val = result["readings"].get("sys")
        dia_val = result["readings"].get("dia")
        result["category"] = database.calculate_category(sys_val, dia_val)

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Base64 inference error")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ============================================================================
# Batch Inference & Processing Endpoints (Sequential CPU execution)
# ============================================================================

@app.post("/api/batch-process")
async def batch_process_endpoint(
    images: List[UploadFile] = File(...),
    variant: str = Form("SMART-BP+"),
    conf_threshold: float = Form(0.2),
    _token: str = Depends(verify_access)
):
    """
    Sequential Batch Inference Endpoint.
    Extracts original photo capture dates (DateTimeOriginal, DateTimeDigitized, DateTime)
    from image EXIF metadata, checks for duplicates via SHA-256 hash, and runs SMART-BP
    inference one-by-one to preserve memory on Render Free instances (512 MB RAM).
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided in batch upload.")

    if len(images) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size limit of {MAX_BATCH_IMAGES} images exceeded. Please upload in smaller batches."
        )

    # Load inference engine once for entire batch
    engine = get_inference_engine(variant=variant)

    results = []
    total_batch_bytes = 0

    count_success = 0
    count_unreadable = 0
    count_duplicates = 0
    count_missing_dates = 0

    for idx, img_file in enumerate(images):
        filename = img_file.filename or f"image_{idx+1}.jpg"
        try:
            image_bytes = await img_file.read()
            if not image_bytes:
                results.append({
                    "batch_index": idx,
                    "filename": filename,
                    "status": "error",
                    "error_message": "Empty file received.",
                    "readings": {"sys": None, "dia": None, "pul": None},
                    "category": "Unknown",
                    "confidence": {},
                    "metadata": {"has_capture_date": False, "display_datetime": "Capture date unavailable", "image_hash": None},
                    "thumbnail_base64": None
                })
                count_unreadable += 1
                continue

            total_batch_bytes += len(image_bytes)
            if total_batch_bytes > MAX_BATCH_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total batch size exceeded {MAX_BATCH_SIZE_BYTES // (1024*1024)}MB limit."
                )

            # 1. Extract EXIF metadata BEFORE any image transformations
            metadata = exif_utils.extract_exif_metadata(image_bytes)
            if not metadata["has_capture_date"]:
                count_missing_dates += 1

            # 2. Check for duplicate image upload via SHA-256 hash
            img_hash = metadata["image_hash"]
            duplicate_record = database.check_duplicate_image(img_hash)
            is_dup = duplicate_record is not None
            if is_dup:
                count_duplicates += 1

            # 3. Create lightweight thumbnail (~1KB)
            thumb_b64 = exif_utils.create_thumbnail_base64(image_bytes)

            # 4. Sequential inference on CPU
            infer_result = engine.predict(
                image_input=image_bytes,
                conf_threshold=conf_threshold,
                generate_annotated_image=False  # Do not build large annotated image for batch to save RAM
            )

            readings = infer_result.get("readings", {})
            conf_info = infer_result.get("confidence", {})
            has_valid_bp = readings.get("sys") is not None and readings.get("dia") is not None

            if is_dup:
                status = "duplicate"
                if has_valid_bp:
                    count_success += 1
                else:
                    count_unreadable += 1
            elif has_valid_bp:
                count_success += 1
                if not metadata["has_capture_date"]:
                    status = "success_date_missing"
                else:
                    status = "success"
            else:
                count_unreadable += 1
                status = "unreadable"

            sys_val = readings.get("sys")
            dia_val = readings.get("dia")
            category = database.calculate_category(sys_val, dia_val)

            results.append({
                "batch_index": idx,
                "filename": filename,
                "status": status,
                "is_duplicate": is_dup,
                "duplicate_info": duplicate_record,
                "readings": readings,
                "confidence": conf_info,
                "category": category,
                "metadata": metadata,
                "thumbnail_base64": thumb_b64,
                "message": infer_result.get("message", "")
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error processing batch item {filename}")
            results.append({
                "batch_index": idx,
                "filename": filename,
                "status": "error",
                "error_message": str(e),
                "readings": {"sys": None, "dia": None, "pul": None},
                "category": "Unknown",
                "confidence": {},
                "metadata": {"has_capture_date": False, "display_datetime": "Capture date unavailable", "image_hash": None},
                "thumbnail_base64": None
            })
            count_unreadable += 1
        finally:
            # Explicit garbage collection after each image to stay inside 512MB RAM
            gc.collect()

    return {
        "status": "success",
        "total_processed": len(images),
        "summary": {
            "total_uploaded": len(images),
            "successful_readings": count_success,
            "unreadable_or_failed": count_unreadable,
            "duplicates_detected": count_duplicates,
            "missing_exif_dates": count_missing_dates
        },
        "items": results
    }


@app.post("/api/batch-confirm")
def batch_confirm_endpoint(
    req: BatchConfirmRequest,
    _token: str = Depends(verify_access)
):
    """
    Explicit batch confirmation endpoint.
    Saves user-reviewed readings to the database with original capture timestamps.
    Requires validated timestamps (from EXIF or user selection).
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="No readings provided to save.")

    items_to_save = []
    for item in req.items:
        ts = (item.timestamp or "").strip()
        if not ts:
            raise HTTPException(
                status_code=400,
                detail=f"Reading from '{item.image_filename or 'batch'}' is missing a capture date/time. Please select a date/time before saving."
            )

        items_to_save.append({
            "sys": item.sys,
            "dia": item.dia,
            "pulse": item.pulse,
            "timestamp": ts,
            "model_variant": item.model_variant,
            "confidence": item.confidence,
            "original_sys": item.original_sys,
            "original_dia": item.original_dia,
            "original_pulse": item.original_pulse,
            "notes": item.notes or "",
            "image_hash": item.image_hash,
            "image_filename": item.image_filename,
            "capture_date_source": item.capture_date_source or "UserConfirmed"
        })

    saved = database.save_batch_readings(items_to_save)
    return {
        "status": "success",
        "message": f"Successfully saved {len(saved)} verified blood pressure readings.",
        "saved_count": len(saved),
        "records": saved
    }


# ============================================================================
# Confirmation & Database Endpoints
# ============================================================================

@app.post("/api/confirm-reading")
def confirm_and_save_reading(
    req: ConfirmReadingRequest,
    _token: str = Depends(verify_access)
):
    """
    Explicit confirmation endpoint: Saves user-verified reading to database.
    Requires authentication. Scoped strictly to the personal user.
    """
    try:
        saved_record = database.save_reading(
            sys=req.sys,
            dia=req.dia,
            pulse=req.pulse,
            model_variant=req.model_variant,
            confidence=req.confidence,
            original_sys=req.original_sys,
            original_dia=req.original_dia,
            original_pulse=req.original_pulse,
            notes=req.notes or "",
            custom_timestamp=req.timestamp,
            image_hash=req.image_hash,
            image_filename=req.image_filename,
            capture_date_source=req.capture_date_source
        )
        return {"status": "success", "message": "Reading saved successfully.", "reading": saved_record}
    except Exception as e:
        logger.exception("Failed to save reading")
        raise HTTPException(status_code=500, detail=f"Database save error: {str(e)}")


@app.get("/api/history")
def get_reading_history(
    limit: int = Query(500, ge=1, le=1000),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    order: str = Query("DESC"),
    _token: str = Depends(verify_access)
):
    """Retrieves blood pressure history with optional date filtering. Requires authentication."""
    try:
        records = database.get_all_readings(
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            order=order
        )
        return {"status": "success", "count": len(records), "readings": records}
    except Exception as e:
        logger.exception("Failed to fetch history")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{reading_id}")
def delete_history_reading(
    reading_id: int,
    _token: str = Depends(verify_access)
):
    """Deletes a record from history. Requires authentication."""
    success = database.delete_reading(reading_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reading not found or unauthorized.")
    return {"status": "success", "message": f"Reading {reading_id} deleted."}


@app.get("/api/stats")
def get_history_stats(_token: str = Depends(verify_access)):
    """Returns analytics and summary metrics. Requires authentication."""
    return database.get_stats()


@app.get("/api/chart-data")
def get_chart_data_endpoint(
    period: Optional[str] = Query("all"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    _token: str = Depends(verify_access)
):
    """
    Returns chronologically sorted (ASC) readings and descriptive summary statistics
    for rendering time-series charts across daily, weekly, monthly, or custom ranges.
    """
    return database.get_chart_data(
        start_date=start_date,
        end_date=end_date,
        period=period
    )


@app.get("/api/sample-image")
def get_sample_image(_token: str = Depends(verify_access)):
    """Returns the pre-generated ground truth test monitor image. Requires authentication."""
    sample_path = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found on disk.")
    return FileResponse(sample_path, media_type="image/png", filename="sample_bp_monitor.png")


# ============================================================================
# Static Files & Frontend App Serving
# ============================================================================

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SMART-BP API is running. UI index.html not yet built."}


def get_local_ip() -> str:
    """Returns the primary local network IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn
    logger.info("Initializing SMART-BP inference engine...")
    get_inference_engine(variant="SMART-BP+")
    database.init_db()

    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    local_ip = get_local_ip()

    print("\n" + "=" * 66)
    print("  BLOOD PRESSURE TRACKER & SMART-BP AI SERVER READY")
    print("=" * 66)
    print(f"  * Local URL  : http://127.0.0.1:{port}/")
    print(f"  * Network URL: http://{local_ip}:{port}/")
    print(f"  * Bound Host : {host}:{port}")

    if IS_AUTO_GENERATED_TOKEN and not os.environ.get("APP_ACCESS_TOKEN"):
        print("\n" + "!" * 66)
        print("  [SECURITY NOTICE] APP_ACCESS_TOKEN was not set in environment.")
        print("  A cryptographically secure session passkey has been generated:")
        print(f"    ACTIVE ACCESS TOKEN: {ACTIVE_ACCESS_TOKEN}")
        print("  Enter this passkey into your browser/mobile app to unlock.")
        print("  Set APP_ACCESS_TOKEN in your environment for a persistent key.")
        print("!" * 66)

    print("=" * 66 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
