"""
SMART-BP / Clifford Lab Blood Pressure Digitization Inference Engine
Based on: "An Open Large-Scale, Real-World Dataset of Blood Pressure Device Images
           and a Benchmark Algorithm for Edge Transcription" (Clifford Lab, 2026)

This module provides a robust, production-ready inference wrapper around the official
pretrained YOLOv8s SMART-BP and SMART-BP+ models.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image
import cv2

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_bp")

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics YOLO is not installed. Please install it in your environment.")
    YOLO = None

# Default paths to model weights inside the project directory (configurable via env vars)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("SMART_BP_MODEL_DIR") or os.path.join(
    PROJECT_ROOT,
    "21269694",
    "cliffordlab",
    "BP_image_digitize-v1.1.0",
    "cliffordlab-BP_image_digitize-e681187",
    "Saved_Models"
)

DEFAULT_WEIGHTS = {
    "SMART-BP+": os.environ.get("SMART_BP_PLUS_WEIGHTS") or os.path.join(MODEL_DIR, "SMART-BP+_Weights.pt"),
    "SMART-BP": os.environ.get("SMART_BP_WEIGHTS") or os.path.join(MODEL_DIR, "SMART-BP_Weights.pt")
}

# In-memory model cache for fast subsequent predictions
_MODEL_CACHE: Dict[str, Any] = {}


# ============================================================================
# Official Clifford Lab Pipeline Classes (SMART-BP benchmark implementation)
# ============================================================================

class Box:
    """
    Represents a single detected bounding box from YOLO output.
    """
    def __init__(self, coordinate: List[float], confidence: float, classId: int, names: Dict[int, str]):
        self.coordinate = coordinate
        self.x1 = float(coordinate[0])  # left
        self.y1 = float(coordinate[1])  # top
        self.x2 = float(coordinate[2])  # right
        self.y2 = float(coordinate[3])  # bottom
        self.confidence = float(confidence)
        self.classId = int(classId)
        self.names = names

    def className(self) -> str:
        """Returns human-readable class label (e.g. '0'-'9' or '10')."""
        return str(self.names.get(self.classId, self.classId))

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def isInside(self, otherBox: "Box") -> bool:
        """
        Returns True if this digit box belongs inside otherBox.
        Allows leading digits (e.g. '1' in '101') that partially extend outside
        the container horizontally, provided vertical overlap and proximity hold.
        """
        x1Intersection = max(otherBox.x1, self.x1)
        y1Intersection = max(otherBox.y1, self.y1)
        x2Intersection = min(otherBox.x2, self.x2)
        y2Intersection = min(otherBox.y2, self.y2)
        widthIntersection = max(0.0, x2Intersection - x1Intersection)
        heightIntersection = max(0.0, y2Intersection - y1Intersection)
        intersection_area = widthIntersection * heightIntersection

        if self.area() <= 0:
            return False

        # 2D area overlap
        if (intersection_area / self.area()) > 0.45:
            return True

        # Vertical overlap check (in case container is clipped horizontally)
        y_overlap_ratio = heightIntersection / self.height() if self.height() > 0 else 0
        self_xc = (self.x1 + self.x2) / 2
        h_near = (otherBox.x1 - self.width() * 0.6) <= self_xc <= (otherBox.x2 + self.width() * 0.6)

        return (y_overlap_ratio > 0.60 and h_near)


class BoxCollection:
    """
    Stores a group of digit bounding boxes corresponding to a single BP value
    (e.g., systolic, diastolic, or pulse).
    """
    def __init__(self):
        self.digits: List[Box] = []  # array of Box objects
        self.value: Optional[int] = None  # reconstructed numeric value

    def deduplicateDigits(self, iou_thresh: float = 0.35):
        """
        Suppresses duplicate bounding boxes detecting the same physical 7-segment digit.
        Keeps the higher-confidence detection.
        """
        non_10 = [d for d in self.digits if d.className() != "10"]
        box_10 = [d for d in self.digits if d.className() == "10"]
        if len(non_10) <= 1:
            return

        sorted_digits = sorted(non_10, key=lambda d: d.confidence, reverse=True)
        kept = []
        for d in sorted_digits:
            is_dup = False
            for k in kept:
                x_inter = max(0.0, min(d.x2, k.x2) - max(d.x1, k.x1))
                min_w = min(d.width(), k.width())
                if min_w > 0 and (x_inter / min_w) > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)

        self.digits = box_10 + sorted(kept, key=lambda d: d.x1)

    def sortDigit(self):
        """Sort digits from left to right based on x-coordinate."""
        n = len(self.digits)
        for i in range(n - 1):
            swapped = False
            for j in range(0, n - i - 1):
                if self.digits[j].x1 > self.digits[j + 1].x1:
                    swapped = True
                    self.digits[j], self.digits[j + 1] = self.digits[j + 1], self.digits[j]
            if not swapped:
                return

    def getValue(self) -> Optional[int]:
        self.deduplicateDigits(iou_thresh=0.35)
        self.sortDigit()
        non_10_digits = [d for d in self.digits if d.className() != "10"]
        # Blood pressure values have 2 or 3 digits (filters clock/time strings with 4 digits and 1-digit noise)
        if len(non_10_digits) < 2 or len(non_10_digits) > 3:
            self.value = None
            return None

        concatDigit = "".join(d.className() for d in non_10_digits)
        if concatDigit != "":
            try:
                self.value = int(concatDigit)
            except ValueError:
                self.value = None
        else:
            self.value = None
        return self.value

    def get10Box(self) -> Optional[Box]:
        for digit in self.digits:
            if digit.className() == "10":
                return digit
        return None


class BPValues:
    """
    Stores extracted BP measurements:
    - Systolic (SYS)
    - Diastolic (DIA)
    - Pulse (PUL)
    """
    def __init__(self):
        self.values: List[BoxCollection] = []

    def sys(self) -> Union[int, str]:
        try:
            val = self.values[0].getValue()
            return val if val is not None else "-"
        except IndexError:
            return "-"

    def dia(self) -> Union[int, str]:
        try:
            val = self.values[1].getValue()
            return val if val is not None else "-"
        except IndexError:
            return "-"

    def pul(self) -> Union[int, str]:
        try:
            val = self.values[2].getValue()
            return val if val is not None else "-"
        except IndexError:
            return "-"

    def identifyValueType(self):
        """
        Assign values purely by vertical position:
          Top    -> SYS
          Middle -> DIA
          Bottom -> PUL
        Works even if only 1 or 2 values are detected.
        """
        if len(self.values) == 0:
            return

        def y_center_of_value(v: BoxCollection) -> float:
            b10 = v.get10Box()
            if b10 is not None:
                return (b10.y1 + b10.y2) / 2.0
            # Fallback: use digit centers if '10' box missing
            ys = []
            for d in v.digits:
                if d.className() != "10":
                    ys.append((d.y1 + d.y2) / 2.0)
            return float(np.mean(ys)) if ys else float("inf")

        # Sort by vertical position (top to bottom)
        self.values.sort(key=y_center_of_value)

    def averageConfidence(self) -> float:
        sumConf = 0.0
        countConf = 0
        for value in self.values:
            for digit in value.digits:
                sumConf += digit.confidence
                countConf += 1
        if countConf == 0:
            return 0.0
        return sumConf / countConf

    def lowestConfidence(self) -> float:
        min_conf = 1.0
        count = 0
        for value in self.values:
            for digit in value.digits:
                count += 1
                if digit.confidence < min_conf:
                    min_conf = digit.confidence
        return min_conf if count > 0 else 0.0

    def highestConfidence(self) -> float:
        max_conf = 0.0
        for value in self.values:
            for digit in value.digits:
                if digit.confidence > max_conf:
                    max_conf = digit.confidence
        return max_conf


# ============================================================================
# Post-Processing Helpers (from SMART-BP benchmark)
# ============================================================================

def value_y_center(v: BoxCollection) -> float:
    b10 = v.get10Box()
    if b10 is not None:
        return float((float(b10.y1) + float(b10.y2)) / 2.0)
    ys = []
    for d in v.digits:
        if d.className() != "10":
            ys.append((float(d.y1) + float(d.y2)) / 2.0)
    return float(np.mean(ys)) if ys else float("inf")


def digit_count(v: BoxCollection) -> int:
    return sum(d.className() != "10" for d in v.digits)


def avg_digit_conf(v: BoxCollection) -> float:
    ds = [d for d in v.digits if d.className() != "10"]
    if not ds:
        return 0.0
    return float(np.mean([float(d.confidence) for d in ds]))


def group_score(v: BoxCollection) -> Tuple[int, float, int]:
    """
    Prefer:
    1) more digits
    2) higher average confidence
    3) larger decoded value
    """
    try:
        val = int(v.getValue() or 0)
    except Exception:
        val = 0
    return (digit_count(v), avg_digit_conf(v), val)


def keep_best_group_per_row(values: List[BoxCollection], y_tol: float = 35.0) -> List[BoxCollection]:
    """
    Collapse duplicate groups detected in the same row.
    """
    if not values:
        return []

    vals = sorted(values, key=value_y_center)
    rows: List[Dict[str, Any]] = []

    for v in vals:
        yc = value_y_center(v)
        placed = False
        for row in rows:
            if abs(yc - row["yc"]) <= y_tol:
                row["items"].append(v)
                row["yc"] = float(np.mean([value_y_center(x) for x in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"yc": yc, "items": [v]})

    picked = []
    for row in sorted(rows, key=lambda r: r["yc"]):
        best = max(row["items"], key=group_score)
        picked.append(best)

    return picked


def _try_fix_group(group: BoxCollection, kind: str) -> bool:
    """
    Validation safeguard: preserves detected digits and does not silently delete them.
    """
    return False


def validate_bp_value(val: Any, kind: str) -> Optional[int]:
    """
    Validates physiological plausibility:
    - SYS: 40 - 260 mmHg
    - DIA: 30 - 200 mmHg
    - PUL: 25 - 240 BPM
    Returns valid integer or None.
    """
    try:
        val = int(val)
    except (ValueError, TypeError):
        return None

    if kind == "sys":
        if 40 <= val <= 260:
            return val
    elif kind == "dia":
        if 30 <= val <= 200:
            return val
    elif kind == "pul":
        if 25 <= val <= 240:
            return val

    return None


def score_measurement(measurement: BPValues) -> float:
    """
    Computes an objective quality score for an orientation extraction.
    Rewards complete readings, physiological plausibility (SYS > DIA), and column alignment.
    """
    sys_raw = measurement.sys()
    dia_raw = measurement.dia()
    pul_raw = measurement.pul()

    sys_val = validate_bp_value(sys_raw, "sys")
    dia_val = validate_bp_value(dia_raw, "dia")
    pul_val = validate_bp_value(pul_raw, "pul")

    valid_count = sum(1 for v in [sys_val, dia_val, pul_val] if v is not None)
    if valid_count == 0:
        return -100.0

    avg_conf = measurement.averageConfidence()
    score = valid_count * 200.0 + avg_conf * 50.0

    if sys_val is not None:
        score += 50.0
    if dia_val is not None:
        score += 50.0
    if pul_val is not None:
        score += 50.0

    if sys_val is not None and dia_val is not None:
        if sys_val > dia_val:
            score += 100.0
        else:
            score -= 200.0  # Inversion penalty

    # Column alignment bonus: check if rows align vertically within display width
    if len(measurement.values) >= 2:
        xs = []
        for v in measurement.values[:3]:
            b10 = v.get10Box()
            if b10:
                xs.append((b10.x1 + b10.x2) / 2)
            elif v.digits:
                non_10 = [d for d in v.digits if d.className() != "10"]
                if non_10:
                    xs.append(float(np.mean([(d.x1 + d.x2)/2 for d in non_10])))
        if xs and (max(xs) - min(xs)) < 250:
            score += 50.0

    return score


# ============================================================================
# Main Inference Engine Class
# ============================================================================

class SmartBPInferenceEngine:
    """
    Production-ready inference wrapper for the SMART-BP / SMART-BP+ model.
    """
    def __init__(self, variant: str = "SMART-BP+", custom_weights_path: Optional[str] = None):
        """
        Initializes the model engine.
        :param variant: 'SMART-BP+' (synthetic + real) or 'SMART-BP' (real only)
        :param custom_weights_path: Optional explicit path to .pt weights file
        """
        self.variant = variant
        self.weights_path = custom_weights_path or DEFAULT_WEIGHTS.get(variant)
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Loads and caches the YOLO model weights on CPU."""
        if not self.weights_path or not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Model weights file not found at: {self.weights_path}. "
                f"Please ensure pretrained weights exist in {MODEL_DIR}."
            )

        if self.weights_path in _MODEL_CACHE:
            logger.info(f"Using cached model for {self.variant}")
            self.model = _MODEL_CACHE[self.weights_path]
            return

        logger.info(f"Loading {self.variant} weights from {self.weights_path} onto CPU...")
        t0 = time.time()
        self.model = YOLO(self.weights_path)
        load_time = (time.time() - t0) * 1000
        logger.info(f"Successfully loaded {self.variant} in {load_time:.1f}ms. Classes: {self.model.names}")
        _MODEL_CACHE[self.weights_path] = self.model

    def _run_single_inference(
        self,
        img_np: np.ndarray,
        conf_threshold: float = 0.2
    ) -> Tuple[BPValues, List[Dict[str, Any]], Any]:
        """
        Executes YOLO prediction and extracts structured BP measurement for a single orientation.
        """
        try:
            results = self.model.predict(source=img_np, conf=conf_threshold, verbose=False)
        except Exception as e:
            logger.exception("Single inference execution error")
            return BPValues(), [], None

        if not results or len(results) == 0:
            return BPValues(), [], None

        res = results[0]
        boxes_data = res.boxes
        if len(boxes_data.xyxy) == 0:
            return BPValues(), [], res

        measurement = BPValues()
        digitBoxes = BoxCollection()
        valueBoxes = BoxCollection()
        raw_detections = []

        for i in range(len(boxes_data.xyxy)):
            coord = [float(c) for c in boxes_data.xyxy[i]]
            conf = float(boxes_data.conf[i])
            cls_idx = int(boxes_data.cls[i])
            box = Box(coord, conf, cls_idx, self.model.names)
            cls_name = box.className()

            raw_detections.append({
                "class_id": cls_idx,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "box": [round(c, 1) for c in coord]
            })

            if cls_name == "10":
                valueBoxes.digits.append(box)
            else:
                digitBoxes.digits.append(box)

        # Match digits inside each '10' container
        for vbox in valueBoxes.digits:
            value = BoxCollection()
            value.digits.append(vbox)
            for dbox in digitBoxes.digits:
                if dbox.isInside(vbox):
                    value.digits.append(dbox)
            # Deduplicate multiple overlapping detections of the same physical digit
            value.deduplicateDigits(iou_thresh=0.35)
            value.sortDigit()
            # Valid BP value has 2 or 3 non-10 digits (filters 4-digit clocks and 1-digit noise)
            non_10_count = sum(d.className() != "10" for d in value.digits)
            if 2 <= non_10_count <= 3:
                decoded = value.getValue()
                if decoded is not None and 25 <= decoded <= 260:
                    measurement.values.append(value)

        # Disambiguate overlapping row detections
        measurement.values = keep_best_group_per_row(measurement.values, y_tol=40.0)
        measurement.identifyValueType()

        # Try fix group if needed
        changed = False
        if len(measurement.values) >= 1:
            changed |= _try_fix_group(measurement.values[0], "sys")
        if len(measurement.values) >= 2:
            changed |= _try_fix_group(measurement.values[1], "dia")
        if len(measurement.values) >= 3:
            changed |= _try_fix_group(measurement.values[2], "pul")
        if changed:
            measurement.identifyValueType()

        return measurement, raw_detections, res

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes],
        conf_threshold: float = 0.2,
        generate_annotated_image: bool = True
    ) -> Dict[str, Any]:
        """
        Runs SMART-BP inference on an image and returns a structured reading report.
        Automatically evaluates candidate orientations (0, 90, 180, 270 degrees)
        for landscape or ambiguous monitor displays using physiological scoring.

        :param image_input: File path, numpy BGR/RGB array, PIL Image, or raw bytes
        :param conf_threshold: YOLO detection confidence threshold (official default: 0.2)
        :param generate_annotated_image: Whether to generate an annotated JPEG (base64)
        :return: Structured result dictionary
        """
        t_start = time.time()

        # Step 1: Decode image input
        img_np, source_desc, err_res = self._preprocess_input(image_input)
        if err_res is not None:
            return err_res

        # Step 2: Multi-Orientation Evaluation Loop
        h, w = img_np.shape[:2]
        is_landscape = (w > h)

        # Candidate 0: Unrotated original
        meas_0, raw_dets_0, res_0 = self._run_single_inference(img_np, conf_threshold)
        score_0 = score_measurement(meas_0)

        s0 = validate_bp_value(meas_0.sys(), "sys")
        d0 = validate_bp_value(meas_0.dia(), "dia")
        p0 = validate_bp_value(meas_0.pul(), "pul")
        n0 = sum(1 for v in [s0, d0, p0] if v is not None)

        best_meas = meas_0
        best_raw_dets = raw_dets_0
        best_res = res_0
        best_rot = 0
        best_score = score_0
        best_img = img_np

        # Fast path: For portrait photos with 3 valid tiers and SYS > DIA with high score, rot=0 is optimal
        skip_rotations = (not is_landscape) and (n0 == 3) and (s0 is not None and d0 is not None and s0 > d0) and (score_0 >= 500.0)

        if not skip_rotations:
            rotations_to_try = [270, 90] if is_landscape else [90, 180, 270]
            for rot in rotations_to_try:
                if rot == 90:
                    rimg = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    rimg = cv2.rotate(img_np, cv2.ROTATE_180)
                elif rot == 270:
                    rimg = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    continue

                meas_r, raw_dets_r, res_r = self._run_single_inference(rimg, conf_threshold)
                score_r = score_measurement(meas_r)

                if score_r > best_score:
                    best_score = score_r
                    best_meas = meas_r
                    best_raw_dets = raw_dets_r
                    best_res = res_r
                    best_rot = rot
                    best_img = rimg

        # Step 3: Validate and finalize readings
        sys_raw = best_meas.sys()
        dia_raw = best_meas.dia()
        pul_raw = best_meas.pul()

        sys_val = validate_bp_value(sys_raw, "sys")
        dia_val = validate_bp_value(dia_raw, "dia")
        pul_val = validate_bp_value(pul_raw, "pul")

        avg_conf = round(best_meas.averageConfidence(), 4)
        min_conf = round(best_meas.lowestConfidence(), 4)
        max_conf = round(best_meas.highestConfidence(), 4)

        # Determine overall status
        detected_count = sum(1 for v in [sys_val, dia_val, pul_val] if v is not None)
        if detected_count == 3:
            status = "success"
            msg = f"Successfully extracted readings: SYS={sys_val}, DIA={dia_val}, PUL={pul_val}"
        elif detected_count > 0:
            status = "partial"
            msg = f"Partially extracted readings: SYS={sys_val or '-'}, DIA={dia_val or '-'}, PUL={pul_val or '-'}"
        else:
            status = "unreadable"
            msg = "Digital readout could not be reliably transcribed into plausible BP values."

        # Step 4: Generate annotated preview if requested
        annotated_b64 = None
        if generate_annotated_image and best_res is not None:
            try:
                import base64
                plot_img = best_res.plot()
                _, buffer = cv2.imencode(".jpg", plot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not encode annotated preview: {e}")

        total_time_ms = round((time.time() - t_start) * 1000, 2)
        rot_desc = f", rot={best_rot}°" if best_rot != 0 else ""
        logger.info(f"[{status.upper()}] {msg} (conf={avg_conf:.2f}{rot_desc}, time={total_time_ms}ms)")

        return {
            "status": status,
            "model_variant": self.variant,
            "model_weights": os.path.basename(self.weights_path),
            "readings": {
                "sys": sys_val,
                "dia": dia_val,
                "pul": pul_val
            },
            "confidence": {
                "average": avg_conf,
                "lowest": min_conf,
                "highest": max_conf
            },
            "detected_values_count": detected_count,
            "applied_rotation": best_rot,
            "detections": best_raw_dets,
            "annotated_image": annotated_b64,
            "message": msg,
            "inference_time_ms": total_time_ms
        }

    def _preprocess_input(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes]
    ) -> Tuple[Optional[np.ndarray], str, Optional[Dict[str, Any]]]:
        """
        Validates and converts image input to a uint8 BGR numpy array.
        """
        source_desc = "unknown"
        try:
            if isinstance(image_input, str):
                source_desc = image_input
                if not os.path.isfile(image_input):
                    return None, source_desc, {
                        "status": "error",
                        "model_variant": self.variant,
                        "readings": {"sys": None, "dia": None, "pul": None},
                        "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                        "message": f"Image file not found: '{image_input}'",
                        "inference_time_ms": 0.0
                    }
                img = cv2.imread(image_input)
                if img is None:
                    return None, source_desc, {
                        "status": "error",
                        "model_variant": self.variant,
                        "readings": {"sys": None, "dia": None, "pul": None},
                        "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                        "message": f"Unable to decode image format: '{image_input}'",
                        "inference_time_ms": 0.0
                    }
                return img, source_desc, None

            elif isinstance(image_input, bytes):
                source_desc = "bytes"
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None, source_desc, {
                        "status": "error",
                        "model_variant": self.variant,
                        "readings": {"sys": None, "dia": None, "pul": None},
                        "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                        "message": "Invalid or corrupt image binary data.",
                        "inference_time_ms": 0.0
                    }
                return img, source_desc, None

            elif isinstance(image_input, Image.Image):
                source_desc = "PIL.Image"
                img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                return img, source_desc, None

            elif isinstance(image_input, np.ndarray):
                source_desc = "numpy.ndarray"
                return image_input, source_desc, None

            else:
                return None, source_desc, {
                    "status": "error",
                    "model_variant": self.variant,
                    "readings": {"sys": None, "dia": None, "pul": None},
                    "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                    "message": f"Unsupported image input type: {type(image_input)}",
                    "inference_time_ms": 0.0
                }

        except Exception as e:
            return None, source_desc, {
                "status": "error",
                "model_variant": self.variant,
                "readings": {"sys": None, "dia": None, "pul": None},
                "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                "message": f"Error decoding image: {str(e)}",
                "inference_time_ms": 0.0
            }


# Singleton accessor
_DEFAULT_ENGINE = None

def get_inference_engine(variant: str = "SMART-BP+") -> SmartBPInferenceEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or _DEFAULT_ENGINE.variant != variant:
        _DEFAULT_ENGINE = SmartBPInferenceEngine(variant=variant)
    return _DEFAULT_ENGINE


def read_bp_image(image_path_or_bytes: Union[str, bytes], variant: str = "SMART-BP+") -> Dict[str, Any]:
    """Convenience function to run BP inference directly."""
    engine = get_inference_engine(variant=variant)
    return engine.predict(image_path_or_bytes)
