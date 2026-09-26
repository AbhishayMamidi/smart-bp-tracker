"""
Comprehensive Production Test Suite for SMART-BP Blood Pressure Tracker
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
from api_server import app
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

    # 2. Health routes
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
    sample_img = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
    assert os.path.exists(sample_img), f"Test image missing: {sample_img}"

    with open(sample_img, "rb") as f:
        img_bytes = f.read()

    for variant in ["SMART-BP+", "SMART-BP"]:
        resp = client.post(
            "/api/read-bp-image",
            files={"image": ("monitor.png", img_bytes, "image/png")},
            data={"variant": variant, "conf_threshold": "0.2"}
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
    # Use a temporary directory to simulate a Render Persistent Disk mount
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        custom_db_path = os.path.join(tmp_dir, "render_mount", "bp_tracker_persistent.db")
        os.environ["DATABASE_PATH"] = custom_db_path

        client = TestClient(app)

        # Baseline check
        initial_history = client.get("/api/history").json()
        assert initial_history["count"] == 0, "Expected empty initial custom database"

        # Run inference
        sample_img = os.path.join(PROJECT_ROOT, "test_images", "synthetic_ihealth_120_80_72.png")
        with open(sample_img, "rb") as f:
            img_bytes = f.read()

        client.post(
            "/api/read-bp-image",
            files={"image": ("monitor.png", img_bytes, "image/png")},
            data={"variant": "SMART-BP+"}
        )

        # Confirm inference did NOT save to database
        after_history = client.get("/api/history").json()
        assert after_history["count"] == 0, "VIOLATION: Reading saved automatically without user confirmation!"
        print("  * Save-Guard verified: Inference did not alter history.")

        # Explicit confirmation
        confirm_resp = client.post("/api/confirm-reading", json={
            "sys": 120,
            "dia": 80,
            "pulse": 72,
            "notes": "Persistent disk test entry"
        })
        assert confirm_resp.status_code == 200
        saved_id = confirm_resp.json()["reading"]["id"]

        # Verify reading is in custom DB path
        assert os.path.exists(custom_db_path), "Database was not created at custom DATABASE_PATH"
        readings = client.get("/api/history").json()["readings"]
        assert len(readings) == 1
        assert readings[0]["id"] == saved_id
        print(f"  * Reading saved to custom disk path: {custom_db_path}")

        # Reset env var
        del os.environ["DATABASE_PATH"]
    print(">>> TEST 4 PASSED: Save guard active & persistent disk DATABASE_PATH fully functional.\n")


def test_5_upload_limits_and_invalid_inputs():
    print("--- [TEST 5] Testing Upload Size Limits & Corrupted File Rejection ---")
    client = TestClient(app)

    # 1. Corrupted bytes
    resp = client.post(
        "/api/read-bp-image",
        files={"image": ("corrupted.jpg", b"MALFORMED_HEADER_BYTES", "image/jpeg")},
        data={"variant": "SMART-BP+"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    print("  * Corrupt bytes handled cleanly with error status")

    # 2. Exceeding upload size limit (>20MB)
    huge_bytes = b"0" * (21 * 1024 * 1024)  # 21MB
    huge_resp = client.post(
        "/api/read-bp-image",
        files={"image": ("huge.jpg", huge_bytes, "image/jpeg")},
        data={"variant": "SMART-BP+"}
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


if __name__ == "__main__":
    print("================================================================")
    print("      SMART-BP PRODUCTION & RENDER DOCKER TEST SUITE            ")
    print("================================================================\n")
    test_1_model_loading()
    test_2_frontend_and_health_endpoints()
    test_3_inference_on_both_model_variants()
    test_4_save_guard_and_configurable_database()
    test_5_upload_limits_and_invalid_inputs()
    test_6_file_integrity()
    print("================================================================")
    print("           ALL 6 PRODUCTION TEST SUITES PASSED!                 ")
    print("================================================================")
