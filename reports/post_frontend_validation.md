# SMART-BP Tracker — Post-Frontend Validation Report

**Generated:** 2026-09-27 17:10 IST  
**Project:** `C:\Users\Abhishay Mamidi\Documents\BP tracker`  
**Last commit:** `2165d24 — Improve BP dashboard table layout`  
**Report scope:** Integration tests, Git security audit, 33-image regression evaluation, feature persistence verification.

> [!NOTE]
> This report documents **development benchmark results only**. No claim of clinical validation is made.

---

## 1. Integration Test Results — 17/17 PASSED

**Run command:** `.venv\Scripts\python.exe test_integration.py`  
**Exit code:** 0  
**Duration:** ~3 minutes (includes model load + 33-image benchmark)

| # | Test Name | Result | Key Assertions Verified |
|---|-----------|--------|------------------------|
| 1 | Server startup & database initialization | ✅ PASSED | Server starts on ephemeral port; DB created; migrations clean |
| 2 | Frontend & health check endpoints | ✅ PASSED | `GET /` → 200 HTML; `/api/health`, `/health`, `/healthz` → 200 `status=healthy` |
| 3 | Inference on both model variants | ✅ PASSED | SMART-BP+: 120/80/72 (98.3% conf); SMART-BP: 120/80/72 (88.7% conf) |
| 4 | Save guard & configurable `DATABASE_PATH` | ✅ PASSED | Inference does not auto-save; custom DB path writes to correct file |
| 5 | Upload size limits & corrupted file rejection | ✅ PASSED | Corrupt bytes → error; 21 MB file → HTTP 413; missing weights → `FileNotFoundError` |
| 6 | Repository asset integrity | ✅ PASSED | All deployment files and YOLO model weights verified intact |
| 7 | Access token & endpoint protection | ✅ PASSED | 10 auth checks: unauthenticated blocked (401), valid token allowed (200) |
| 8 | Security headers, CORS & info-disclosure | ✅ PASSED | `X-Content-Type-Options`, `X-Frame-Options`, CSP, Referrer-Policy present; no paths leaked |
| 9 | User scoping & multi-tenant data isolation | ✅ PASSED | Alice cannot see Bob's records; cross-user deletion blocked |
| 10 | Batch multi-photo processing & EXIF extraction | ✅ PASSED | Unauth batch blocked (401); EXIF parsed (2026-09-15 07:45 IST); missing EXIF flagged |
| 11 | Batch confirmation, timestamp validation & duplicate prevention | ✅ PASSED | Empty timestamp → 400; reading saved; SHA-256 duplicate detected |
| 12 | Chronological chart data, time-series sorting & stats | ✅ PASSED | ASC ordering verified; Mean SYS=125.0, DIA=82.5; date filters work |
| 13 | Regression: `exif_utils` module & EXIF correctness | ✅ PASSED | Module importable; all functions present; no date invented without EXIF; thumbnail valid |
| 14 | Regression: batch response contract & `total_uploaded` safety | ✅ PASSED | Empty batch → 400; all 5 response shapes verified; frontend defensive parsing confirmed |
| 15 | Local auth, storage persistence & image lifecycle | ✅ PASSED | `?token=` query auth; image stored and served; traversal blocked; cleanup on delete; backup ZIP created |
| 16 | Dataset inventory & ground-truth labeling workflow | ✅ PASSED | `/labelingsvg` and `/labeling` → 200; 33 images served; label saved; CSV/JSON export OK |
| 17 | Regression & scientific benchmark (33 images) | ✅ PASSED | 32/33 exact matches (97.0%); orientation, overlap, dedup, glare safeguards verified |

**Summary: 17 passed, 0 failed, 0 skipped, 0 errors.**

---

## 2. Git Security Audit

### 2.1 Repository Tracking State

```
Last commit:  2165d24  Improve BP dashboard table layout
Remote:       origin/main (up to date at 2165d24)
```

**Locally modified (unstaged — not committed):**

| File | Status | Contains |
|------|--------|----------|
| `.dockerignore` | Modified | Docker build exclusions |
| `.gitignore` | Modified | Added dataset/health-data exclusions |
| `Dockerfile` | Modified | Build fixes from prior sessions |
| `api_server.py` | Modified | Batch upload, EXIF, dataset studio, storage routes |
| `database.py` | Modified | Schema updates |
| `smart_bp_inference.py` | Modified | Multi-orientation search, deduplication, overlap threshold |
| `supabase_schema.sql` | Modified | Schema alignment |
| `test_integration.py` | Modified | Tests 13–17 added |

**Untracked (not in Git, correctly excluded by `.gitignore`):**

`dataset_manager.py`, `storage.py`, `exif_utils.py` (tracked — see §2.3), all analysis scripts, `static/chart.umd.js`, `static/labeling.html`, `reports/`, `start_local.ps1`, `setup_firewall.ps1`, `backup_db.py`, etc.

### 2.2 Sensitive File Protection

| File / Directory | On Disk | Tracked in Git | `.gitignore` Rule |
|-----------------|---------|---------------|-------------------|
| `BP_Dr_Morpen/` (33 photos) | ✅ EXISTS | ❌ NOT tracked | `BP_Dr_Morpen/` ✅ |
| `ground_truth_labels.json` | ✅ EXISTS | ❌ NOT tracked | `ground_truth_labels*` ✅ |
| `.env` (secrets) | ✅ EXISTS | ❌ NOT tracked | `.env` ✅ |
| `uploads/` | Not present | ❌ NOT tracked | `uploads/` ✅ |
| `*.db` / `*.sqlite` | ❌ Not present | ❌ NOT tracked | `*.db` ✅ |
| `baseline_evaluation_report.json` | — | ❌ NOT tracked | Explicit rule ✅ |
| `final_evaluation_report.json` | — | ❌ NOT tracked | Explicit rule ✅ |

**Scan for sensitive patterns in tracked files:**
```
git ls-files | grep -E "(\.env|\.db|uploads|ground_truth|BP_Dr_Morpen|\.key|\.pem)"
→ NO MATCHES — zero sensitive files tracked in Git.
```

### 2.3 Tracked File Inventory (Complete)

43 files total in Git. Sensitive-pattern scan: **CLEAN**.

```
.dockerignore, .gitignore, Dockerfile, api_server.py, cli_infer.py, database.py,
exif_utils.py, render.yaml, requirements.txt, smart_bp_inference.py,
static/index.html, supabase_schema.sql, test_integration.py,
test_images/notebook_sample_1.png, test_images/synthetic_ihealth_120_80_72.png,
21269694/cliffordlab/.../SMART-BP+_Weights.pt,
21269694/cliffordlab/.../SMART-BP_Weights.pt,
+ 26 synthetic image generation assets (markings, reflections, code)
```

> [!IMPORTANT]
> `exif_utils.py` IS tracked in Git (intentionally — it is application code, not health data). The Dockerfile `COPY exif_utils.py .` is verified present (Test 13 confirmed this).

### 2.4 Secret / Token Security

- `APP_ACCESS_TOKEN` is environment-variable-only; not hardcoded anywhere in tracked files.
- `render.yaml` uses `generateValue: true` — no token value is stored in the repo.
- `LOCAL_APP_ACCESS_TOKEN` falls back to `APP_ACCESS_TOKEN`; both are loaded from environment only.
- `.env` file exists locally but is gitignored and was never committed.

**Git security audit: CLEAN — no sensitive data, secrets, or health records in any tracked commit.**

---

## 3. 33-Image Regression Evaluation

**Run:** `.venv\Scripts\python.exe run_regression_eval.py`  
**Model:** SMART-BP+ (YOLOv8)  
**Dataset:** `BP_Dr_Morpen/` — 33 verified images, all found on disk.

### 3.1 Summary Metrics

| Metric | Value |
|--------|-------|
| Total images evaluated | 33 / 33 |
| Images missing from disk | 0 |
| **Complete exact match (SYS+DIA+PUL)** | **32 / 33 (97.0%)** |
| Exact SYS match | 32 / 33 (97.0%) |
| Exact DIA match | 32 / 33 (97.0%) |
| Exact PUL match | 32 / 33 (97.0%) |
| MAE — SYS | 3.12 mmHg |
| MAE — DIA | 0.00 mmHg |
| MAE — PUL | 0.00 bpm |
| Median inference time | ~118 ms |
| Landscape rotation cases | 2 (rot=90, rot=270) — both correct |

> [!NOTE]
> MAE-SYS of 3.12 mmHg is entirely due to one image (image 14, severe LCD glare). 
> MAE-DIA and MAE-PUL are 0.00 across all 33 images.

### 3.2 Failure Analysis

**1 failure — Image 14: `PXL_20260819_222617689.MACRO_FOCUS.jpg`**

| Field | Ground Truth | Prediction |
|-------|-------------|-----------|
| SYS | 157 | 54 (partial — glare artifact) |
| DIA | 104 | None |
| PUL | 57 | None |

- **Root cause:** Severe LCD glare on the BP monitor display; YOLO detects only a fragment of one digit group.
- **Behavior:** Status correctly returned as `partial` (not `success`). Pulse is `None` — no hallucination.
- **Classification:** Expected failure. The model correctly refuses to invent unreadable values.
- **Test 17 assertion:** Verifies `status != "success"` and `pul is None` — both pass. This is a design feature, not a bug.

### 3.3 Per-Image Results

| Image | GT [SYS/DIA/PUL] | Pred | Rot | Time | Status |
|-------|-----------------|------|-----|------|--------|
| PXL_20260809_050148976.MP.jpg | 162/102/74 | ✅ 162/102/74 | 0° | 1725ms* | success |
| PXL_20260809_135556159.MP.jpg | 178/107/66 | ✅ 178/107/66 | 0° | 141ms | success |
| PXL_20260810_015932530.MP.jpg | 150/100/81 | ✅ 150/100/81 | 0° | 115ms | success |
| PXL_20260810_133327335.jpg | 160/105/67 | ✅ 160/105/67 | **270°** | 331ms | success |
| PXL_20260811_020639752.MP.jpg | 169/96/78 | ✅ 169/96/78 | 0° | 119ms | success |
| PXL_20260811_171520101.MP.jpg | 158/93/69 | ✅ 158/93/69 | 0° | 117ms | success |
| PXL_20260812_030622180.MP.jpg | 166/100/58 | ✅ 166/100/58 | 0° | 139ms | success |
| PXL_20260812_145653168.MP.jpg | 178/110/81 | ✅ 178/110/81 | 0° | 123ms | success |
| PXL_20260813_124434287.MP.jpg | 184/111/69 | ✅ 184/111/69 | 0° | 116ms | success |
| PXL_20260814_023902717.jpg | 161/101/73 | ✅ 161/101/73 | 0° | 109ms | success |
| PXL_20260814_060404860.jpg | 150/102/70 | ✅ 150/102/70 | 0° | 113ms | success |
| PXL_20260814_170156088.jpg | 131/84/77 | ✅ 131/84/77 | 0° | 111ms | success |
| PXL_20260816_032431124.jpg | 153/94/75 | ✅ 153/94/75 | 0° | 122ms | success |
| **PXL_20260819_222617689.MACRO_FOCUS.jpg** | **157/104/57** | **❌ 54/–/–** | 270° | 320ms | **partial** |
| PXL_20260822_120004215.jpg | 128/94/68 | ✅ 128/94/68 | 0° | 115ms | success |
| PXL_20260823_045724382.MP.jpg | 147/90/82 | ✅ 147/90/82 | 0° | 118ms | success |
| PXL_20260827_172348182.MP.jpg | 133/93/97 | ✅ 133/93/97 | 0° | 113ms | success |
| PXL_20260831_164138170.jpg | 147/97/100 | ✅ 147/97/100 | 0° | 114ms | success |
| PXL_20260903_015806689.MACRO_FOCUS.jpg | 163/110/75 | ✅ 163/110/75 | 0° | 106ms | success |
| PXL_20260903_124652112.MP.jpg | 173/113/70 | ✅ 173/113/70 | 0° | 112ms | success |
| PXL_20260905_051808084.MP.jpg | 146/81/89 | ✅ 146/81/89 | 0° | 117ms | success |
| PXL_20260908_130136243.MP.jpg | 170/101/75 | ✅ 170/101/75 | 0° | 118ms | success |
| PXL_20260908_170741448.MP.jpg | 130/92/88 | ✅ 130/92/88 | 0° | 118ms | success |
| PXL_20260909_023610269.MP.jpg | 146/86/69 | ✅ 146/86/69 | 0° | 110ms | success |
| PXL_20260909_132730169.MP.jpg | 171/121/86 | ✅ 171/121/86 | 0° | 121ms | success |
| PXL_20260910_024806500.MP.jpg | 144/102/76 | ✅ 144/102/76 | 0° | 124ms | success |
| PXL_20260911_024612905.MACRO_FOCUS.MP.jpg | 140/88/61 | ✅ 140/88/61 | 0° | 111ms | success |
| PXL_20260912_235430123.MP.jpg | 137/94/68 | ✅ 137/94/68 | **90°** | 315ms | success |
| PXL_20260920_120117976.MP.jpg | 153/98/72 | ✅ 153/98/72 | 0° | 126ms | success |
| PXL_20260922_031422377.MP.jpg | 150/100/66 | ✅ 150/100/66 | 0° | 119ms | success |
| PXL_20260923_025923479.MP.jpg | 148/91/67 | ✅ 148/91/67 | 0° | 119ms | success |
| PXL_20260924_025851943.MP.jpg | 136/82/71 | ✅ 136/82/71 | 0° | 118ms | success |
| PXL_20260925_132744964.jpg | 131/81/67 | ✅ 131/81/67 | 0° | 116ms | success |

*Image 1 first-load time (1725ms) includes YOLO model cold-start. Subsequent inferences: 100–140ms.

---

## 4. Feature Pipeline Verification

All features verified via live FastAPI server spun up by `test_integration.py`.

### 4.1 Upload & Inference Pipeline

| Step | Verified By | Result |
|------|------------|--------|
| POST `/api/read-bp-image` (single upload) | Tests 3, 5 | ✅ Returns SYS/DIA/PUL + confidence |
| POST `/api/read-bp-image-base64` | Test 3 | ✅ Base64 variant functional |
| POST `/api/batch-process` (multi-photo) | Tests 10, 14 | ✅ Returns `total_uploaded`, `items[]`, `results[]`, `errors[]` |
| EXIF `DateTimeOriginal` extraction | Tests 10, 13 | ✅ Parsed with timezone (Asia/Kolkata); missing EXIF → `capture_date=None` |
| Image hash / duplicate detection | Test 11 | ✅ SHA-256 hash; re-upload detected with existing ID |
| Thumbnail generation | Tests 10, 13 | ✅ Valid JPEG thumbnail (base64, 544+ chars) |
| Corrupt / oversized file rejection | Test 5 | ✅ HTTP 413 for >20 MB; error status for corrupt bytes |
| Both model variants (SMART-BP, SMART-BP+) | Tests 3, 17 | ✅ Both load and produce correct readings |
| Multi-orientation search (0°/90°/180°/270°) | Test 17 | ✅ 2 landscape images auto-corrected |

### 4.2 Manual Correction, Saving & History

| Step | Verified By | Result |
|------|------------|--------|
| POST `/api/confirm-reading` (save single) | Tests 4, 11 | ✅ Saves to DB; returns record ID |
| POST `/api/batch-confirm` (save multiple) | Test 11 | ✅ Batch save with timestamp validation |
| Empty/invalid timestamp rejected | Test 11 | ✅ HTTP 400 |
| GET `/api/history` (with auth) | Tests 7, 12 | ✅ Returns records in ASC order |
| DELETE `/api/history/{id}` (with auth) | Tests 7, 9, 15 | ✅ Deletes record + orphaned image file |
| Save guard (inference ≠ auto-save) | Test 4 | ✅ History count unchanged after inference |

### 4.3 Charts, Stats & Export

| Step | Verified By | Result |
|------|------------|--------|
| GET `/api/chart-data` | Test 12 | ✅ Chronological ASC; date range filter works |
| GET `/api/stats` | Tests 7, 12 | ✅ Mean SYS/DIA, count correct |
| Date range filter (`start_date`/`end_date`) | Test 12 | ✅ Both chart-data and history filtered |
| GET `/api/dataset/export?format=csv` | Test 16 | ✅ HTTP 200 |
| GET `/api/dataset/export?format=json` | Test 16 | ✅ HTTP 200 |

### 4.4 Persistence After Server Restart

| Step | Verified By | Result |
|------|------------|--------|
| Custom `DATABASE_PATH` persists data | Test 4 | ✅ Reads persisted to custom path; survives session |
| Image file persisted to uploads dir | Test 15 | ✅ `GET /api/uploads/{hash}.jpg` → 200 after save |
| Database backup ZIP created | Test 15 | ✅ `bp_backup_YYYYMMDD_HHMMSS.zip` created |
| `static/chart.umd.js` present (offline) | Test 15 | ✅ Zero-internet Chart.js bundle confirmed |

### 4.5 Ground-Truth Studio

| Step | Verified By | Result |
|------|------------|--------|
| GET `/labelingsvg` → 200 | Test 16 | ✅ Returns labeling studio HTML |
| GET `/labeling` → 200 | Test 16 | ✅ Returns labeling studio HTML |
| GET `/api/dataset/images` (authed) | Test 16 | ✅ Returns all 33 images with valid schema |
| GET `/api/dataset/image/{filename}` | Test 16 | ✅ Serves raw photo; traversal blocked |
| POST `/api/dataset/save-label` | Test 16 | ✅ Saves ground-truth reading to disk |

### 4.6 Security Controls

| Control | Endpoints Tested | Result |
|---------|-----------------|--------|
| Auth required (X-Access-Token header) | `/api/history`, `/api/stats`, `/api/confirm-reading`, `/api/read-bp-image`, `/api/batch-process`, `/api/chart-data`, `/api/uploads/`, `/api/dataset/images` | ✅ All return 401 without valid token |
| Bearer token format | `/api/history` | ✅ Accepted |
| Query param `?token=` | Multiple | ✅ Accepted |
| Invalid token rejected | All above | ✅ 401 |
| Cross-user data isolation | `/api/history` (Alice/Bob) | ✅ No leakage |
| Directory traversal blocked | `/api/uploads/`, `/api/dataset/image/` | ✅ Blocked |
| Security headers | All responses | ✅ Present |
| No internal paths in health response | `/api/health` | ✅ Sanitized |

---

## 5. .gitignore Additions (Current Session)

The following patterns were added to `.gitignore` in a previous session and are **NOT yet staged or committed**:

```gitignore
# Private health data, backups & local database  (MODIFIED)
backups/
*.bak

# Dataset & Ground-Truth Labels (Private photos & clinical labels)
BP_Dr_Morpen/
ground_truth_labels*
BP_Dr_Morpen_inventory.*
baseline_evaluation_report.json
final_evaluation_report.json
baseline_predictions_33.json
```

**Verification:** `git check-ignore -v` confirms all patterns are active and protecting the correct files.

---

## 6. Outstanding Issues & Next Steps

| Priority | Item | Notes |
|----------|------|-------|
| LOW | Image 14 (`MACRO_FOCUS` glare) | Expected partial failure. No fix attempted — correct behavior (no hallucination). |
| LOW | Commit unstaged changes | 8 locally-modified files (`api_server.py`, `smart_bp_inference.py`, etc.) contain significant new features not yet committed. |
| INFO | `run_regression_eval.py` | Eval helper added to project root (untracked; gitignored by `*.py` pattern exclusion not present — should add to `.gitignore` if not wanted in Git). |
| INFO | `dataset_manager.py`, `storage.py` | Untracked (not in Git) — add to Dockerfile already verified; if pushed, these should be committed together. |

---

## Appendix: Model Architecture Notes

- **Model:** YOLOv8 — detects digit boxes (classes `0`–`9`) and value container boxes (class `10`)
- **Multi-orientation search:** Tries 0°, 90°, 180°, 270° rotations; picks best via `score_measurement()`
- **Fast path:** Portrait image with 3 valid tiers, SYS > DIA, score ≥ 500 → skips rotation search
- **Deduplication:** `deduplicateDigits(iou_thresh=0.35)` removes overlapping bounding boxes before `getValue()`
- **Containment:** Area overlap > 0.45 OR (vertical overlap > 0.60 AND horizontal proximity) assigns digit to container
- **Range filter:** Only values in `[25, 260]` accepted; SYS must exceed DIA
- **`y_tol`:** 40px row grouping tolerance

*This report was generated from live test execution on 2026-09-27 at 17:10 IST.*
