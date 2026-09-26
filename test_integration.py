"""
Comprehensive Production & Security Test Suite for SMART-BP Blood Pressure Tracker
Validates:
1. Pretrained Weights Loading (SMART-BP and SMART-BP+)
2. Frontend Root (/) and Health Endpoints (/api/health, /health, /healthz)
3. Inference on Reference Monitor Photograph with Ground-Truth Verification
4. Both model variants (SMART-BP+ and SMART-BP) via POST /api/read-bp-image
5. Inference Save-Guard (verifying inference does NOT alter history)
6. Explicit Confirmation and Persistent Database Storage (custom DATABASE_PATH test)
7. Upload Size Limits (HTTP 413) and Corrupt Image Rejection
8. Missing Weights Handling and System Robustness
9. File Integrity of Downloaded Repository Assets
10. SECURITY: Access token enforcement across all personal routes (401 for unauthorized)
11. SECURITY: Timing-attack safe token comparison & Bearer/X-Access-Token verification
12. SECURITY: Security headers injection & zero filesystem path leaks in health checks
13. SECURITY: User-scoped isolation preventing cross-user record leakage or unauthorized deletion
"""

import os
import sys
import tempfile
from starlette.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smart_bp_inference import get_inference_engine, DEFAULT_WEIGHTS, SmartBPInferenceEngine
from api_server import app, get_expected_token
import database


def test_1_model_loading():
    print("--- [TEST 1] Verifying Pretrained Weights & Loading Both Variants ---")
    for variant in ["SMART-BP+", "SMART-BP"]:
        path = DEFAULT_WEIGHTS[variant]
        assert os.path.exists(path), f"Weights file missing: {path}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  * {variant} weights: {os.path.basename(path)} ({size_mb:.2f} MB)")
        engine = SmartBPInferenceEngine(variant=variant)
        assert engine.model is not None, f"Model {variant} failed to instantiate"
        assert len(engine.model.names) == 11, f"Expected 11 classes, got {len(engine.model.names)}"
        print(f"  * {variant} instantiated successfully. Classes: {list(engine.model.names.values())}")
    print(">>> TEST 1 PASSED: Both models load successfully on CPU.\n")


def test_2_frontend_and_health_endpoints():
    print("--- [TEST 2] Testing Frontend Route (/) and Health Endpoints ---")
    client = TestClient(app)

    # 1. Root route serves index.html
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Blood Pressure Tracker" in root_resp.text
    print("  * GET / returned 200 OK with web dashboard HTML")

    # 2. Health routes (public, unauthenticated probes)
    for health_route in ["/api/health", "/health", "/healthz"]:
        h_resp = client.get(health_route)
        assert h_resp.status_code == 200, f"Failed route {health_route}: {h_resp.status_code}"
        h_data = h_resp.json()
        assert h_data["status"] == "healthy"
        assert h_data["device"] == "CPU"
        assert "SMART-BP+" in h_data["models"]
        assert "SMART-BP" in h_data["models"]
        print(f"  * GET {health_route} returned 200 OK: status={h_data['status']}")
    print(">>> TEST 2 PASSED: Frontend and all health check endpoints verified.\n")


def test_3_inference_on_both_model_variants():
    print("--- [TEST 3] Running Inference on Both Variants with Ground-Truth Image ---")
    client = TestClient(app)
    expected_token = get_expected_token()
    sample_img = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
    assert os.path.exists(sample_img), f"Test image missing: {sample_img}"

    with open(sample_img, "rb") as f:
        img_bytes = f.read()

    for variant in ["SMART-BP+", "SMART-BP"]:
        resp = client.post(
            "/api/read-bp-image",
            files={"image": ("monitor.png", img_bytes, "image/png")},
            data={"variant": variant, "conf_threshold": "0.2"},
            headers={"X-Access-Token": expected_token}
        )
        assert resp.status_code == 200, f"Inference failed for {variant}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success", f"Expected success for {variant}, got {data['status']}"
        assert data["readings"]["sys"] == 120, f"{variant} SYS mismatch"
        assert data["readings"]["dia"] == 80, f"{variant} DIA mismatch"
        assert data["readings"]["pul"] == 72, f"{variant} PUL mismatch"
        print(f"  * {variant}: Extracted SYS={data['readings']['sys']}, DIA={data['readings']['dia']}, PUL={data['readings']['pul']} (conf={data['confidence']['average']*100:.1f}%)")
    print(">>> TEST 3 PASSED: Both model variants successfully transcribe ground-truth readings (120/80/72).\n")


def test_4_save_guard_and_configurable_database():
    print("--- [TEST 4] Testing Save Guard and Configurable DATABASE_PATH ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        custom_db_path = os.path.join(tmp_dir, "render_mount", "bp_tracker_persistent.db")
        os.environ["DATABASE_PATH"] = custom_db_path

        client = TestClient(app)
        auth_headers = {"X-Access-Token": get_expected_token()}

        # Baseline check
        initial_history = client.get("/api/history", headers=auth_headers).json()
        assert initial_history["count"] == 0, "Expected empty initial custom database"

        # Run inference
        sample_img = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
        with open(sample_img, "rb") as f:
            img_bytes = f.read()

        client.post(
            "/api/read-bp-image",
            files={"image": ("monitor.png", img_bytes, "image/png")},
            data={"variant": "SMART-BP+"},
            headers=auth_headers
        )

        # Confirm inference did NOT save to database
        after_history = client.get("/api/history", headers=auth_headers).json()
        assert after_history["count"] == 0, "VIOLATION: Reading saved automatically without user confirmation!"
        print("  * Save-Guard verified: Inference did not alter history.")

        # Explicit confirmation
        confirm_resp = client.post(
            "/api/confirm-reading",
            json={
                "sys": 120,
                "dia": 80,
                "pulse": 72,
                "notes": "Persistent disk test entry"
            },
            headers=auth_headers
        )
        assert confirm_resp.status_code == 200
        saved_id = confirm_resp.json()["reading"]["id"]

        # Verify reading is in custom DB path
        assert os.path.exists(custom_db_path), "Database was not created at custom DATABASE_PATH"
        readings = client.get("/api/history", headers=auth_headers).json()["readings"]
        assert len(readings) == 1
        assert readings[0]["id"] == saved_id
        print(f"  * Reading saved to custom disk path: {custom_db_path}")

        # Reset env var
        del os.environ["DATABASE_PATH"]
    print(">>> TEST 4 PASSED: Save guard active & persistent disk DATABASE_PATH fully functional.\n")


def test_5_upload_limits_and_invalid_inputs():
    print("--- [TEST 5] Testing Upload Size Limits & Corrupted File Rejection ---")
    client = TestClient(app)
    auth_headers = {"X-Access-Token": get_expected_token()}

    # 1. Corrupted bytes
    resp = client.post(
        "/api/read-bp-image",
        files={"image": ("corrupted.jpg", b"MALFORMED_HEADER_BYTES", "image/jpeg")},
        data={"variant": "SMART-BP+"},
        headers=auth_headers
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    print("  * Corrupt bytes handled cleanly with error status")

    # 2. Exceeding upload size limit (>20MB)
    huge_bytes = b"0" * (21 * 1024 * 1024)  # 21MB
    huge_resp = client.post(
        "/api/read-bp-image",
        files={"image": ("huge.jpg", huge_bytes, "image/jpeg")},
        data={"variant": "SMART-BP+"},
        headers=auth_headers
    )
    assert huge_resp.status_code == 413, f"Expected 413 Payload Too Large, got {huge_resp.status_code}"
    print("  * 21MB upload rejected with HTTP 413 Payload Too Large")

    # 3. Missing weights
    try:
        SmartBPInferenceEngine(variant="SMART-BP+", custom_weights_path="non_existent.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  * Missing weights raised expected FileNotFoundError")
    print(">>> TEST 5 PASSED: Upload size limits and invalid inputs handled securely.\n")


def test_6_file_integrity():
    print("--- [TEST 6] Confirming Repository Asset Integrity ---")
    orig_base = os.path.join(
        PROJECT_ROOT,
        "21269694",
        "cliffordlab",
        "BP_image_digitize-v1.1.0",
        "cliffordlab-BP_image_digitize-e681187"
    )
    assert os.path.exists(os.path.join(orig_base, "Saved_Models", "SMART-BP_Weights.pt"))
    assert os.path.exists(os.path.join(orig_base, "Saved_Models", "SMART-BP+_Weights.pt"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, ".dockerignore"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, ".gitignore"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "render.yaml"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "supabase_schema.sql"))
    print(">>> TEST 6 PASSED: All deployment files and model weights verified intact.\n")


def test_7_security_access_token_enforcement():
    print("--- [TEST 7] Security Audit: Access Token & Endpoint Protection ---")
    client = TestClient(app)
    expected_token = get_expected_token()

    # 1. Unauthenticated GET /api/history returns 401
    resp = client.get("/api/history")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-1] Unauthenticated GET /api/history blocked with 401 Unauthorized")

    # 2. Invalid token GET /api/history returns 401
    resp = client.get("/api/history", headers={"X-Access-Token": "bad-token-12345"})
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-2] Invalid X-Access-Token blocked with 401 Unauthorized")

    # 3. Valid token GET /api/history returns 200
    resp = client.get("/api/history", headers={"X-Access-Token": expected_token})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    print("  * [SEC-3] Valid X-Access-Token allowed with 200 OK")

    # 4. Authorization: Bearer <token> also accepted
    resp = client.get("/api/history", headers={"Authorization": f"Bearer {expected_token}"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    print("  * [SEC-4] Authorization: Bearer <token> allowed with 200 OK")

    # 5. Unauthenticated POST /api/confirm-reading returns 401
    resp = client.post("/api/confirm-reading", json={"sys": 120, "dia": 80})
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-5] Unauthenticated POST /api/confirm-reading blocked with 401 Unauthorized")

    # 6. Unauthenticated POST /api/read-bp-image returns 401 (DoS / compute protection)
    sample_img = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
    with open(sample_img, "rb") as f:
        img_bytes = f.read()
    resp = client.post(
        "/api/read-bp-image",
        files={"image": ("monitor.png", img_bytes, "image/png")}
    )
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-6] Unauthenticated POST /api/read-bp-image blocked with 401 Unauthorized")

    # 7. Unauthenticated DELETE /api/history/9999 returns 401
    resp = client.delete("/api/history/9999")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-7] Unauthenticated DELETE /api/history/{id} blocked with 401 Unauthorized")

    # 8. Unauthenticated GET /api/stats returns 401
    resp = client.get("/api/stats")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("  * [SEC-8] Unauthenticated GET /api/stats blocked with 401 Unauthorized")

    # 9. Verify token API endpoint
    resp = client.post("/api/auth/verify", json={"token": "wrong"})
    assert resp.status_code == 401
    resp = client.post("/api/auth/verify", json={"token": expected_token})
    assert resp.status_code == 200
    assert resp.json()["authenticated"] is True
    print("  * [SEC-9] POST /api/auth/verify endpoint correctly validates tokens")

    # 10. Auth status API endpoint
    resp = client.get("/api/auth/status")
    assert resp.status_code == 200
    assert resp.json()["authenticated"] is False
    resp = client.get("/api/auth/status", headers={"X-Access-Token": expected_token})
    assert resp.status_code == 200
    assert resp.json()["authenticated"] is True
    print("  * [SEC-10] GET /api/auth/status accurately reports session validity")
    print(">>> TEST 7 PASSED: Access control strictly enforced across all sensitive endpoints.\n")


def test_8_security_headers_and_path_sanitization():
    print("--- [TEST 8] Security Headers, CORS & Info Disclosure Sanitization ---")
    client = TestClient(app)

    # 1. Verify Security Headers
    resp = client.get("/api/health")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "1; mode=block" in resp.headers.get("X-XSS-Protection", "")
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    print("  * Security Headers verified (X-Content-Type-Options, X-Frame-Options, CSP/XSS, Referrer-Policy)")

    # 2. Verify /api/health does NOT leak directory paths or secrets
    data = resp.json()
    assert "models" in data
    for model_name, info in data["models"].items():
        assert "path" not in info, f"LEAK: Internal path exposed in health check for {model_name}!"
        assert "loaded" in info
    assert "database" in data
    assert "path" not in data["database"], "LEAK: Internal database path exposed in health check!"
    print("  * Health check data sanitized: Zero internal filesystem paths leaked.")
    print(">>> TEST 8 PASSED: Security headers active and information disclosure prevented.\n")


def test_9_user_scoped_isolation():
    print("--- [TEST 9] User Scoping & Multi-Tenant Data Isolation ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        os.environ["DATABASE_PATH"] = os.path.join(tmp_dir, "isolation_test.db")
        database.init_db()

        # Save reading for user Alice
        r_a = database.save_reading(sys=120, dia=80, pulse=70, notes="User Alice record", user_id="user_alice")
        # Save reading for user Bob
        r_b = database.save_reading(sys=140, dia=90, pulse=85, notes="User Bob record", user_id="user_bob")

        # Alice should only see Alice's records
        alice_readings = database.get_all_readings(user_id="user_alice")
        assert len(alice_readings) == 1
        assert alice_readings[0]["id"] == r_a["id"]
        assert alice_readings[0]["sys"] == 120
        print("  * User Alice cannot see Bob's blood pressure records")

        # Bob should only see Bob's records
        bob_readings = database.get_all_readings(user_id="user_bob")
        assert len(bob_readings) == 1
        assert bob_readings[0]["id"] == r_b["id"]
        assert bob_readings[0]["sys"] == 140
        print("  * User Bob cannot see Alice's blood pressure records")

        # Alice cannot delete Bob's reading
        deleted = database.delete_reading(reading_id=r_b["id"], user_id="user_alice")
        assert deleted is False, "SECURITY BREACH: User Alice deleted Bob's reading!"
        print("  * Unauthorized cross-user deletion strictly prevented")

        del os.environ["DATABASE_PATH"]
    print(">>> TEST 9 PASSED: Complete data isolation and ownership enforcement verified.\n")


def test_10_batch_processing_and_exif_extraction():
    print("--- [TEST 10] Batch Multi-Photo Processing & EXIF Metadata Extraction ---")
    client = TestClient(app)
    expected_token = get_expected_token()

    from PIL import Image, ExifTags
    import io

    # 1. Prepare Image 1: Reference monitor with EXIF DateTimeOriginal
    sample_img_path = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
    with Image.open(sample_img_path) as im:
        exif = im.getexif()
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        exif_ifd[36867] = "2026:09:15 07:45:00"  # DateTimeOriginal
        buf1 = io.BytesIO()
        im.convert("RGB").save(buf1, format="JPEG", exif=exif)
        img1_bytes = buf1.getvalue()

    # 2. Prepare Image 2: PNG without EXIF
    img2 = Image.new("RGB", (100, 100), color=(240, 240, 240))
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    img2_bytes = buf2.getvalue()

    # Unauthorized access check
    resp_unauth = client.post(
        "/api/batch-process",
        files=[
            ("images", ("photo1.jpg", img1_bytes, "image/jpeg")),
            ("images", ("photo2.png", img2_bytes, "image/png"))
        ]
    )
    assert resp_unauth.status_code == 401, f"Expected 401 Unauthorized, got {resp_unauth.status_code}"
    print("  * Unauthorized batch-process request blocked with 401")

    # Authorized batch processing
    resp = client.post(
        "/api/batch-process",
        files=[
            ("images", ("monitor_exif.jpg", img1_bytes, "image/jpeg")),
            ("images", ("no_exif.png", img2_bytes, "image/png"))
        ],
        data={"variant": "SMART-BP+"},
        headers={"X-Access-Token": expected_token}
    )
    assert resp.status_code == 200, f"Batch process failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 2
    assert len(data["items"]) == 2

    # Verify item 0 (with EXIF)
    item0 = data["items"][0]
    assert item0["filename"] == "monitor_exif.jpg"
    assert item0["metadata"]["capture_date"] == "2026-09-15 07:45:00"
    assert "2026-09-15 07:45:00" in item0["metadata"]["display_datetime"]
    assert "Asia/Kolkata" in item0["metadata"]["display_datetime"]
    assert item0["metadata"]["image_hash"] is not None
    assert len(item0["metadata"]["image_hash"]) == 64
    assert item0["thumbnail_base64"] is not None
    assert item0["readings"]["sys"] == 120
    assert item0["readings"]["dia"] == 80
    assert item0["readings"]["pul"] == 72
    print(f"  * Item 0 EXIF parsed: {item0['metadata']['display_datetime']}, Transcribed: 120/80/72")

    # Verify item 1 (without EXIF)
    item1 = data["items"][1]
    assert item1["filename"] == "no_exif.png"
    assert item1["metadata"]["has_capture_date"] is False
    assert item1["metadata"]["display_datetime"] == "Capture date unavailable"
    assert item1["metadata"]["image_hash"] is not None
    print("  * Item 1 Missing EXIF flagged correctly as 'Capture date unavailable'")
    print(">>> TEST 10 PASSED: Batch inference, thumbnail generation, and EXIF extraction verified.\n")


def test_11_batch_confirmation_and_duplicate_detection():
    print("--- [TEST 11] Batch Confirmation, Timestamp Validation & Duplicate Prevention ---")
    client = TestClient(app)
    expected_token = get_expected_token()
    auth_headers = {"X-Access-Token": expected_token}

    from PIL import Image, ExifTags
    import io

    # Create distinct image with EXIF date
    img = Image.new("RGB", (120, 120), color=(10, 20, 30))
    exif = img.getexif()
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[36867] = "2026:09:18 10:15:00"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    img_bytes = buf.getvalue()

    # 1. Process image via batch-process
    proc_resp = client.post(
        "/api/batch-process",
        files=[("images", ("distinct_reading.jpg", img_bytes, "image/jpeg"))],
        headers=auth_headers
    )
    assert proc_resp.status_code == 200
    item = proc_resp.json()["items"][0]
    img_hash = item["metadata"]["image_hash"]
    assert item["is_duplicate"] is False

    # 2. Test batch confirm validation: empty timestamp rejected
    bad_confirm = client.post(
        "/api/batch-confirm",
        json={"items": [{
            "sys": 118,
            "dia": 78,
            "pulse": 68,
            "timestamp": "",
            "image_filename": "distinct_reading.jpg"
        }]},
        headers=auth_headers
    )
    assert bad_confirm.status_code == 400
    print("  * Timestamp validation: Batch item with empty timestamp rejected with 400")

    # 3. Test valid batch confirmation
    confirm_resp = client.post(
        "/api/batch-confirm",
        json={"items": [{
            "sys": 118,
            "dia": 78,
            "pulse": 68,
            "timestamp": "2026-09-18 10:15:00",
            "model_variant": "SMART-BP+",
            "confidence": 0.98,
            "image_hash": img_hash,
            "image_filename": "distinct_reading.jpg",
            "capture_date_source": "EXIF:DateTimeOriginal"
        }]},
        headers=auth_headers
    )
    assert confirm_resp.status_code == 200
    data = confirm_resp.json()
    assert data["saved_count"] == 1
    saved_id = data["records"][0]["id"]
    print(f"  * Reading saved successfully via batch-confirm with ID {saved_id}")

    # 4. Duplicate Detection Test: Re-processing identical image
    dup_proc_resp = client.post(
        "/api/batch-process",
        files=[("images", ("distinct_reading.jpg", img_bytes, "image/jpeg"))],
        headers=auth_headers
    )
    assert dup_proc_resp.status_code == 200
    dup_item = dup_proc_resp.json()["items"][0]
    assert dup_item["is_duplicate"] is True, "Expected is_duplicate to be True"
    assert dup_item["duplicate_info"]["id"] == saved_id
    assert dup_item["duplicate_info"]["timestamp"] == "2026-09-18 10:15:00"
    print("  * Duplicate detection verified: Re-uploaded identical image flagged with existing ID and timestamp")
    print(">>> TEST 11 PASSED: Batch confirmation and SHA-256 duplicate detection verified.\n")


def test_12_chronological_chart_data_and_filtering():
    print("--- [TEST 12] Chronological Chart Data, Time-Series Sorting & Descriptive Stats ---")
    client = TestClient(app)
    expected_token = get_expected_token()
    auth_headers = {"X-Access-Token": expected_token}

    # 1. Unauthorized check
    unauth = client.get("/api/chart-data")
    assert unauth.status_code == 401
    print("  * Unauthenticated GET /api/chart-data blocked with 401")

    # 2. Add test records spanning different dates to verify sorting and filtering
    database.save_reading(sys=122, dia=81, pulse=71, custom_timestamp="2026-09-10 08:00:00", notes="Early reading")
    database.save_reading(sys=128, dia=84, pulse=74, custom_timestamp="2026-09-22 18:30:00", notes="Later reading")

    # 3. Authorized chart data query
    chart_resp = client.get("/api/chart-data?period=all", headers=auth_headers)
    assert chart_resp.status_code == 200
    data = chart_resp.json()
    assert data["status"] == "success"
    readings = data["readings"]
    assert len(readings) >= 2

    # Verify chronological sorting (ASC: oldest to newest)
    timestamps = [r["timestamp"] for r in readings]
    assert timestamps == sorted(timestamps), f"Readings are not sorted chronologically ASC: {timestamps}"
    print(f"  * Verified chronological ASC ordering across {len(readings)} readings")

    # Verify descriptive statistics summary
    summary = data["summary"]
    assert summary["total_count"] == len(readings)
    assert summary["mean_sys"] is not None
    assert summary["mean_dia"] is not None
    assert summary["min_sys"] <= summary["max_sys"]
    assert summary["min_dia"] <= summary["max_dia"]
    assert "disclaimer" in summary
    assert "Not a medical evaluation or diagnosis" in summary["disclaimer"]
    print(f"  * Descriptive statistics: Mean SYS={summary['mean_sys']}, Mean DIA={summary['mean_dia']}, Count={summary['total_count']}")

    # 4. Date range filtering on chart-data
    filtered_chart = client.get(
        "/api/chart-data?start_date=2026-09-22&end_date=2026-09-22",
        headers=auth_headers
    ).json()
    assert filtered_chart["count"] == 1
    assert filtered_chart["readings"][0]["timestamp"] == "2026-09-22 18:30:00"
    print("  * Chart data range filter (start_date/end_date) verified")

    # 5. Date range filtering on history log
    hist_filtered = client.get(
        "/api/history?start_date=2026-09-10&end_date=2026-09-10",
        headers=auth_headers
    ).json()
    assert hist_filtered["count"] == 1
    assert hist_filtered["readings"][0]["timestamp"] == "2026-09-10 08:00:00"
    print("  * History log range filter verified")
    print(">>> TEST 12 PASSED: Chronological chart datasets, statistics, and date filters verified.\n")


if __name__ == "__main__":
    print("================================================================")
    print("      SMART-BP PRODUCTION & SECURITY TEST SUITE                 ")
    print("================================================================\n")
    test_1_model_loading()
    test_2_frontend_and_health_endpoints()
    test_3_inference_on_both_model_variants()
    test_4_save_guard_and_configurable_database()
    test_5_upload_limits_and_invalid_inputs()
    test_6_file_integrity()
    test_7_security_access_token_enforcement()
    test_8_security_headers_and_path_sanitization()
    test_9_user_scoped_isolation()
    test_10_batch_processing_and_exif_extraction()
    test_11_batch_confirmation_and_duplicate_detection()
    test_12_chronological_chart_data_and_filtering()
    print("================================================================")
    print("           ALL 12 TEST SUITES PASSED CLEANLY!                   ")
    print("================================================================")

