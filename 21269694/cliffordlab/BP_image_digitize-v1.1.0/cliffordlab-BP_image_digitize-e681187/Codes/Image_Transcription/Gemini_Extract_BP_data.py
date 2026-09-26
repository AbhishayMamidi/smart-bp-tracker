
__author__ = "B. D. Clifford"
__copyright__ = "Copyright 2026, safe+natal"
__license__ = "BSD 3.0"
__version__ = "1.0.0"

import os
import csv
import glob
import json
import time
from PIL import Image
from pydantic import BaseModel  
from google import genai

# This file uses Gemini to parse 7-segment displays on automatic blood pressure devices
# to identify SBP, DBP, pulse, artifacts and PII. Do not use with PHI unless you are using
# an approved and regulated service (e.g., one that is HIPAA-compliant).

# =====================================================================
# 1. SETUP CONFIGURATION 
# =====================================================================
# Generate API key and insert it here -- This is one of two places the user must update
API_KEY = "???????"  

# This is the directory in which all your images are stored. The code will parse every 
# image in this directory with the extensions listed in image_extensions below
# This is the other place the user must update
IMAGE_DIR = r"/path/to/your/images"

# This is the output file: 
OUTPUT_CSV = "blood_pressure_report.csv" 

# Initialize the Gemini Client
client = genai.Client(api_key=API_KEY)

# Collect files and eliminate duplicates caused by Windows case-insensitivity
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
raw_file_list = []
for ext in image_extensions:
    raw_file_list.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

# Converting to a set drops duplicates, sorted() keeps them in order
image_files = sorted(list(set(raw_file_list)))

if not image_files:
    print(f"Python cannot find any matching images at: {IMAGE_DIR}")
    exit()

total_images = len(image_files)
print(f"Success! Found {total_images} UNIQUE images in your Train folder (Duplicates removed).")
print(f"Running Gemini 3.0 Flash Preview with 4s safety pacing.")
print(f"Writing data directly to '{OUTPUT_CSV}'. Tracking progress...\n")

# =====================================================================
# 2. DEFINING THE SCHEMA (NO DEFAULTS / NO OPTIONS)
# =====================================================================
# By making everything a strict str or bool without default assignments,
# Pydantic strips out the "default: null" tags that trigger the 400 error.
class BloodPressureData(BaseModel):
    time: str
    sbp: str
    dbp: str
    pulse: str
    readable: bool
    human_id_detected: bool
    notes: str

# Prompt text containing your updated background text check constraints
prompt = """
Analyze this image of a digital blood pressure monitor screen. 
Extract the values and map them into the requested fields.

Rules:
1. Extract the time (HH:MM format string, or "N/A" if not visible), sbp (Systolic string, e.g. "120" or "N/A"), dbp (Diastolic string), and pulse (string).
2. If heavy glare, reflection, blur, or clipping cuts off an LCD segment or makes any number ambiguous, set readable to false.
3. Check the edges and background. If a person's face, clear fingerprint, name tag, or distinct identifying markings/tattoos are visible, or if there is any printed or written text visible in the background, set human_id_detected to true.
4. Provide a brief sentence in notes if the image is marked unreadable or if human identification/background text is found.
"""

dataset_rows = []

# =====================================================================
# 3. LIVE PROCESSING LOOP
# =====================================================================
for index, img_path in enumerate(image_files, start=1):
    img_name = os.path.basename(img_path)
    
    max_retries = 5
    backoff_delay = 20
    data = None
    
    for attempt in range(1, max_retries + 1):
        print(f"\r [{index}/{total_images}] Analyzing: {img_name} (Attempt {attempt})...", end="", flush=True)
        
        try:
            raw_image = Image.open(img_path)
            
            # Passing configurations via direct dictionary parameters ensures 
            # absolute structural safety with older preview versions of the engine.
            response = client.models.generate_content(
                model='gemini-3-flash-preview', 
                contents=[raw_image, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": BloodPressureData,
                    "temperature": 0.1
                }
            )
            
            data = json.loads(response.text.strip())
            print("Done", flush=True)
            break
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries:
                    print(f"Rate limit hit. Cooling down for {backoff_delay}s...", end="", flush=True)
                    time.sleep(backoff_delay)
                    backoff_delay += 10  
                    continue
                else:
                    print("Failed! Retries exhausted.", flush=True)
                    err_reason = "Rate Limit Exhausted"
            else:
                print(f"Failed! Error: {error_msg[:40]}", flush=True)
                err_reason = f"Error: {error_msg[:30]}"
                break

    # =====================================================================
    # 4. DATA ROW CAPTURE
    # =====================================================================
    if data is None:
        row_dict = {
            "Image Name": img_name, "Time": "Error", "SBP (Systolic)": "Error",
            "DBP (Diastolic)": "Error", "Pulse": "Error", "Human ID?": "Unknown",
            "Readable?": "No", "Notes / Reason": err_reason
        }
    else:
        row_dict = {
            "Image Name": img_name,
            "Time": data.get("time") or "N/A",
            "SBP (Systolic)": data.get("sbp") or "N/A",
            "DBP (Diastolic)": data.get("dbp") or "N/A",
            "Pulse": data.get("pulse") or "N/A",
            "Human ID?": "DETECTED" if data.get("human_id_detected") else "Clear",
            "Readable?": "Yes" if data.get("readable") else "Unreadable",
            "Notes / Reason": data.get("notes") or "-"
        }
    
    dataset_rows.append(row_dict)
    time.sleep(4) 

# =====================================================================
# 5. AUTOMATIC CSV WRITER GENERATION
# =====================================================================
fieldnames = ["Image Name", "Time", "SBP (Systolic)", "DBP (Diastolic)", "Pulse", "Human ID?", "Readable?", "Notes / Reason"]

try:
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  
        writer.writerows(dataset_rows)  
        
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE!")
    print(f"Report successfully saved to spreadsheet: {os.path.abspath(OUTPUT_CSV)}")
    print("="*60)

except Exception as csv_err:
    print(f"\n Could not save CSV file. Is the spreadsheet open in Excel right now? Details: {csv_err}")