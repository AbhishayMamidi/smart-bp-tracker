"""
Dataset & Ground-Truth Management Module for SMART-BP Tracker
Provides safe image access, inventory indexing, and ground-truth label persistence
for the BP_Dr_Morpen monitor dataset.
"""

import os
import json
import csv
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "BP_Dr_Morpen")
GT_JSON_FILE = os.path.join(PROJECT_ROOT, "ground_truth_labels.json")
GT_CSV_FILE = os.path.join(PROJECT_ROOT, "ground_truth_labels.csv")

def ensure_initialized():
    """Ensures ground truth files exist and are up to date."""
    if not os.path.exists(GT_JSON_FILE) or not os.path.exists(GT_CSV_FILE):
        import init_ground_truth
        init_ground_truth.build_ground_truth_template()

def get_dataset_image_path(filename: str) -> Optional[str]:
    """
    Safely resolves path to an image inside BP_Dr_Morpen folder.
    Prevents directory traversal attacks.
    """
    safe_name = os.path.basename(filename)
    full_path = os.path.abspath(os.path.join(DATASET_DIR, safe_name))
    dataset_abs = os.path.abspath(DATASET_DIR)
    
    if not full_path.startswith(dataset_abs):
        return None
    if not os.path.isfile(full_path):
        return None
    return full_path

def get_dataset_images() -> List[Dict[str, Any]]:
    """Returns the list of all images with metadata, candidates, and verified labels."""
    ensure_initialized()
    with open(GT_JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_image_label(
    filename: str,
    sys_val: int,
    dia_val: int,
    pul_val: Optional[int],
    verified: bool = True,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Atomically updates a ground-truth entry in both JSON and CSV files.
    """
    ensure_initialized()
    with open(GT_JSON_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    target_entry = None
    for r in records:
        if r["filename"] == filename:
            r["sys"] = sys_val
            r["dia"] = dia_val
            r["pul"] = pul_val
            r["verified"] = verified
            r["notes"] = notes
            target_entry = r
            break

    if not target_entry:
        raise ValueError(f"Image '{filename}' not found in dataset inventory.")

    # Write JSON atomically
    temp_json = GT_JSON_FILE + ".tmp"
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(temp_json, GT_JSON_FILE)

    # Write CSV atomically
    temp_csv = GT_CSV_FILE + ".tmp"
    with open(temp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "filename", "capture_date", "width", "height", "orientation",
            "candidate_sys", "candidate_dia", "candidate_pul", "candidate_conf",
            "sys", "dia", "pul", "verified", "notes"
        ])
        for r in records:
            writer.writerow([
                r["index"], r["filename"], r["capture_date"], r["width"], r["height"], r["orientation"],
                r["candidate_sys"], r["candidate_dia"], r["candidate_pul"], r["candidate_conf"],
                r["sys"] if r["sys"] is not None else "",
                r["dia"] if r["dia"] is not None else "",
                r["pul"] if r["pul"] is not None else "",
                "YES" if r.get("verified") else "NO",
                r.get("notes", "")
            ])
    os.replace(temp_csv, GT_CSV_FILE)

    return target_entry

def get_accuracy_metrics(current_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes exact field-level and complete-reading accuracy against verified ground truth.
    :param current_predictions: map of {filename: {"sys": int, "dia": int, "pul": int}}
    """
    images = get_dataset_images()
    verified_images = [img for img in images if img.get("verified")]
    
    if not verified_images:
        return {
            "total_verified": 0,
            "message": "No verified ground truth labels available yet."
        }

    sys_correct = 0
    dia_correct = 0
    pul_correct = 0
    complete_correct = 0
    total = len(verified_images)

    evaluations = []
    for img in verified_images:
        fname = img["filename"]
        gt_sys = img["sys"]
        gt_dia = img["dia"]
        gt_pul = img["pul"]

        pred = current_predictions.get(fname, {})
        p_sys = pred.get("sys")
        p_dia = pred.get("dia")
        p_pul = pred.get("pul")

        s_ok = (p_sys == gt_sys)
        d_ok = (p_dia == gt_dia)
        p_ok = (p_pul == gt_pul) if gt_pul is not None else (p_pul is None)
        all_ok = s_ok and d_ok and p_ok

        if s_ok: sys_correct += 1
        if d_ok: dia_correct += 1
        if p_ok: pul_correct += 1
        if all_ok: complete_correct += 1

        evaluations.append({
            "filename": fname,
            "ground_truth": {"sys": gt_sys, "dia": gt_dia, "pul": gt_pul},
            "predicted": {"sys": p_sys, "dia": p_dia, "pul": p_pul},
            "sys_correct": s_ok,
            "dia_correct": d_ok,
            "pul_correct": p_ok,
            "all_correct": all_ok
        })

    return {
        "total_verified": total,
        "complete_reading_accuracy": round(complete_correct / total * 100, 1),
        "sys_accuracy": round(sys_correct / total * 100, 1),
        "dia_accuracy": round(dia_correct / total * 100, 1),
        "pul_accuracy": round(pul_correct / total * 100, 1),
        "evaluations": evaluations
    }
