"""
FastAPI Server for Blood Pressure Tracker & SMART-BP Inference Engine.
Production-ready configuration for Render Docker deployment and local development.
"""

import os
import sys
import base64
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smart_bp_inference import get_inference_engine, DEFAULT_WEIGHTS
import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bp_api")

app = FastAPI(
    title="Blood Pressure Tracker & SMART-BP Digitization API",
    description="Automated digital blood pressure monitor digitization using YOLOv8 SMART-BP.",
    version="1.2.0"
)

# Configurable CORS for production deployment
cors_env = os.environ.get("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum image upload size (default 20MB)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 20)) * 1024 * 1024

# Optional security token for personal privacy on public deployments
APP_ACCESS_TOKEN = os.environ.get("APP_ACCESS_TOKEN", "").strip()


def verify_access(token_header: Optional[str] = None):
    """
    If APP_ACCESS_TOKEN is configured in environment, verifies incoming header.
    If not set, allows open access (with explicit documentation of privacy risk).
    """
    if APP_ACCESS_TOKEN and token_header != APP_ACCESS_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid X-Access-Token header required for personal health records."
        )


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


class Base64ImageRequest(BaseModel):
    image_base64: str
    variant: Optional[str] = "SMART-BP+"
    conf_threshold: Optional[float] = 0.2


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/api/health")
@app.get("/health")
@app.get("/healthz")
def health_check():
    """
    Returns engine health, CPU device status, weights availability,
    and database configuration. Compatible with Docker HEALTHCHECK and cloud runners.
    """
    weights_status = {
        name: {
            "path": path,
            "exists": os.path.exists(path),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else 0
        }
        for name, path in DEFAULT_WEIGHTS.items()
    }
    all_exist = all(w["exists"] for w in weights_status.values())

    return {
        "status": "healthy" if all_exist else "degraded",
        "device": "CPU",
        "models": weights_status,
        "default_variant": "SMART-BP+",
        "database": {
            "type": "Supabase (cloud)" if database._supabase_client else "SQLite",
            "path": database.get_db_path() if not database._supabase_client else "Supabase remote"
        },
        "auth_enabled": bool(APP_ACCESS_TOKEN)
    }


# ============================================================================
# Inference & Reading Endpoints
# ============================================================================

@app.post("/api/read-bp-image")
async def read_bp_image_endpoint(
    image: Optional[UploadFile] = File(None),
    variant: str = Form("SMART-BP+"),
    conf_threshold: float = Form(0.2)
):
    """
    Inference endpoint: Upload a BP monitor photograph.
    Runs SMART-BP model inference and returns the extracted readings.
    NOTE: Readings are NOT automatically saved; user confirmation is required!
    """
    if image is None:
        raise HTTPException(status_code=400, detail="No image file provided.")

    # Validate content type if supplied
    if image.content_type and not (image.content_type.startswith("image/") or image.content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image.content_type}'. Must be an image.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file received.")

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum upload limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.")

        engine = get_inference_engine(variant=variant)
        result = engine.predict(
            image_input=image_bytes,
            conf_threshold=conf_threshold,
            generate_annotated_image=True
        )

        # Add AHA Reference Category for UI presentation
        sys_val = result["readings"].get("sys")
        dia_val = result["readings"].get("dia")
        result["category"] = database.calculate_category(sys_val, dia_val)

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
def read_bp_image_base64_endpoint(req: Base64ImageRequest):
    """
    Inference endpoint accepting base64 encoded data URI (useful for mobile camera frames).
    """
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


@app.post("/api/confirm-reading")
def confirm_and_save_reading(
    req: ConfirmReadingRequest,
    x_access_token: Optional[str] = Header(None)
):
    """
    Explicit confirmation endpoint: Saves user-verified reading to database.
    """
    verify_access(x_access_token)
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
            custom_timestamp=req.timestamp
        )
        return {"status": "success", "message": "Reading saved successfully.", "reading": saved_record}
    except Exception as e:
        logger.exception("Failed to save reading")
        raise HTTPException(status_code=500, detail=f"Database save error: {str(e)}")


@app.get("/api/history")
def get_reading_history(
    limit: int = Query(50, ge=1, le=500),
    x_access_token: Optional[str] = Header(None)
):
    """Retrieves blood pressure history."""
    verify_access(x_access_token)
    try:
        records = database.get_all_readings(limit=limit)
        return {"status": "success", "count": len(records), "readings": records}
    except Exception as e:
        logger.exception("Failed to fetch history")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{reading_id}")
def delete_history_reading(
    reading_id: int,
    x_access_token: Optional[str] = Header(None)
):
    """Deletes a record from history."""
    verify_access(x_access_token)
    success = database.delete_reading(reading_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reading not found.")
    return {"status": "success", "message": f"Reading {reading_id} deleted."}


@app.get("/api/stats")
def get_history_stats(x_access_token: Optional[str] = Header(None)):
    """Returns analytics and summary metrics."""
    verify_access(x_access_token)
    return database.get_stats()


@app.get("/api/sample-image")
def get_sample_image():
    """Returns the pre-generated ground truth test monitor image."""
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
    # Warm up engine on startup
    logger.info("Initializing SMART-BP inference engine...")
    get_inference_engine(variant="SMART-BP+")
    database.init_db()

    # Dynamic port configuration: defaults to $PORT (Render standard: 10000) or 8000 locally
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")

    local_ip = get_local_ip()
    print("\n" + "=" * 62)
    print("  BLOOD PRESSURE TRACKER & SMART-BP AI SERVER READY")
    print("=" * 62)
    print(f"  * Local URL  : http://127.0.0.1:{port}/")
    print(f"  * Network URL: http://{local_ip}:{port}/")
    print(f"  * Bound Host : {host}:{port}")
    print("=" * 62 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
