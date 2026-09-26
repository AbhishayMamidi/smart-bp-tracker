#!/usr/bin/env python3
"""
CLI Test Interface for SMART-BP Blood Pressure Image Transcription
Usage:
    python cli_infer.py <path_to_image> [--variant SMART-BP+|SMART-BP] [--conf 0.2] [--save-annotated output.jpg] [--json]
"""

import sys
import os
import argparse
import json

# Ensure project directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from smart_bp_inference import get_inference_engine


def classify_bp(sys_val, dia_val):
    """
    Standard AHA/ACC Blood Pressure Category Reference (Educational / Classification only).
    Does NOT provide medical diagnosis or treatment recommendations.
    """
    if sys_val is None or dia_val is None:
        return "Incomplete measurement"
    if sys_val < 120 and dia_val < 80:
        return "Normal"
    elif 120 <= sys_val <= 129 and dia_val < 80:
        return "Elevated"
    elif (130 <= sys_val <= 139) or (80 <= dia_val <= 89):
        return "Hypertension Stage 1"
    elif (sys_val >= 140) or (dia_val >= 90):
        return "Hypertension Stage 2"
    return "Measurement Recorded"


def main():
    parser = argparse.ArgumentParser(
        description="SMART-BP Blood Pressure Monitor Image Transcription CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_infer.py test_images/sample_bp.jpg
  python cli_infer.py test_images/synthetic_ihealth_120_80_72.png --variant SMART-BP
  python cli_infer.py photo.jpg --save-annotated detection_result.jpg
        """
    )
    parser.add_argument("image_path", help="Path to digital blood pressure monitor photograph")
    parser.add_argument(
        "--variant", "-v",
        choices=["SMART-BP+", "SMART-BP"],
        default="SMART-BP+",
        help="Model variant: SMART-BP+ (default, trained on synthetic+real) or SMART-BP (real only)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.2,
        help="Detection confidence threshold (default: 0.2)"
    )
    parser.add_argument(
        "--save-annotated", "-o",
        type=str,
        default=None,
        help="Optional path to save detection-annotated image"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON result instead of human-readable report"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Specified image file does not exist: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.variant} model and running inference on '{args.image_path}'...")
    try:
        engine = get_inference_engine(variant=args.variant)
        result = engine.predict(
            image_input=args.image_path,
            conf_threshold=args.conf,
            generate_annotated_image=bool(args.save_annotated)
        )
    except Exception as e:
        print(f"Inference initialization failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Save annotated preview if requested
    if args.save_annotated and result.get("annotated_image"):
        try:
            import base64
            img_data = base64.b64decode(result["annotated_image"])
            with open(args.save_annotated, "wb") as f:
                f.write(img_data)
            print(f"Saved annotated detection image to: {args.save_annotated}")
        except Exception as e:
            print(f"Warning: Could not save annotated image: {e}", file=sys.stderr)

    if args.json:
        # Exclude bulky base64 from stdout JSON
        clean_result = {k: v for k, v in result.items() if k != "annotated_image"}
        print(json.dumps(clean_result, indent=2))
        return

    # Formatted Human-Readable Output
    status = result["status"]
    readings = result["readings"]
    conf = result["confidence"]
    sys_val = readings.get("sys")
    dia_val = readings.get("dia")
    pul_val = readings.get("pul")

    bp_category = classify_bp(sys_val, dia_val)

    print("\n" + "=" * 55)
    print("  SMART-BP BLOOD PRESSURE DIGITIZATION REPORT")
    print("=" * 55)
    print(f"  Model Variant  : {result['model_variant']} ({result.get('model_weights', '')})")
    print(f"  Inference Time : {result.get('inference_time_ms', 0):.1f} ms (CPU)")
    print(f"  Overall Status : {status.upper()}")
    print("-" * 55)
    print("  EXTRACTED READINGS:")
    print(f"    * Systolic  (SYS) : {sys_val if sys_val is not None else '[UNREADABLE]'} mmHg")
    print(f"    * Diastolic (DIA) : {dia_val if dia_val is not None else '[UNREADABLE]'} mmHg")
    print(f"    * Pulse     (PUL) : {pul_val if pul_val is not None else '[UNREADABLE]'} BPM")
    print("-" * 55)
    print(f"  AHA Reference Category : {bp_category}")
    print(f"  Confidence             : Avg {conf['average'] * 100:.1f}%, Min {conf['lowest'] * 100:.1f}%, Max {conf['highest'] * 100:.1f}%")
    print(f"  Detected Box Count     : {len(result.get('detections', []))} elements")
    print(f"  Diagnostic Message     : {result.get('message', '')}")
    print("=" * 55)
    print("  NOTE: For personal tracking only. Not a medical diagnosis.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
