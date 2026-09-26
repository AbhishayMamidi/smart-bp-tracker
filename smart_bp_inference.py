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
        Returns True if this box overlaps significantly (> 75%) inside otherBox.
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
        return (intersection_area / self.area() > 0.75)


class BoxCollection:
    """
    Stores a group of digit bounding boxes corresponding to a single BP value
    (e.g., systolic, diastolic, or pulse).
    """
    def __init__(self):
        self.digits: List[Box] = []  # array of Box objects
        self.value: Optional[int] = None  # reconstructed numeric value

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
        self.sortDigit()
        concatDigit = ""
        for digit in self.digits:
            if digit.className() != "10":
                concatDigit += digit.className()
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
    Post-processing step to fix implausible BP values.
    Iteratively removes lowest-confidence digit if outside physiological bounds.
    """
    digits = [d for d in group.digits if d.className() != "10"]
    if len(digits) < 3:
        return False

    val = group.getValue()
    if val is None:
        return False

    if kind == "sys":
        bad = (val < 50) or (val > 240)
    elif kind == "dia":
        bad = (val < 30) or (val > 180)
    else:  # pulse
        bad = (val < 25) or (val > 200)

    if not bad:
        return False

    worst = min(digits, key=lambda d: d.confidence)
    group.digits.remove(worst)
    return True


def validate_bp_value(val: Any, kind: str) -> Optional[int]:
    """
    Validates physiological plausibility:
    - SYS: 50 - 240 mmHg
    - DIA: 30 - 180 mmHg
    - PUL: 25 - 200 BPM
    Returns valid integer or None.
    """
    try:
        val = int(val)
    except (ValueError, TypeError):
        return None

    if kind == "sys":
        if 50 <= val <= 240:
            return val
    elif kind == "dia":
        if 30 <= val <= 180:
            return val
    elif kind == "pul":
        if 25 <= val <= 200:
            return val

    return None


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

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes],
        conf_threshold: float = 0.2,
        generate_annotated_image: bool = True
    ) -> Dict[str, Any]:
        """
        Runs SMART-BP inference on an image and returns a structured reading report.

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

        # Step 2: Run official YOLOv8 object detection
        try:
            results = self.model.predict(source=img_np, conf=conf_threshold, verbose=False)
        except Exception as e:
            logger.exception("Inference prediction error")
            return {
                "status": "error",
                "model_variant": self.variant,
                "readings": {"sys": None, "dia": None, "pul": None},
                "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                "message": f"Inference execution failed: {str(e)}",
                "inference_time_ms": round((time.time() - t_start) * 1000, 2)
            }

        if not results or len(results) == 0:
            return {
                "status": "unreadable",
                "model_variant": self.variant,
                "readings": {"sys": None, "dia": None, "pul": None},
                "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                "message": "No blood pressure display detected in image.",
                "inference_time_ms": round((time.time() - t_start) * 1000, 2)
            }

        res = results[0]
        boxes_data = res.boxes

        if len(boxes_data.xyxy) == 0:
            return {
                "status": "unreadable",
                "model_variant": self.variant,
                "readings": {"sys": None, "dia": None, "pul": None},
                "confidence": {"average": 0.0, "lowest": 0.0, "highest": 0.0},
                "message": "No digits or measurement containers identified.",
                "inference_time_ms": round((time.time() - t_start) * 1000, 2)
            }

        # Step 3: Group digits into value boxes using official SMART-BP logic
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
            foundCount = 0
            value = BoxCollection()
            value.digits.append(vbox)
            for dbox in digitBoxes.digits:
                if dbox.isInside(vbox):
                    value.digits.append(dbox)
                    foundCount += 1
                if foundCount == 3:
                    break  # Blood pressure value has at most 3 digits
            value.sortDigit()
            # Only accept containers with at least 2 non-10 digits
            if sum(d.className() != "10" for d in value.digits) >= 2:
                measurement.values.append(value)

        # Disambiguate overlapping row detections
        measurement.values = keep_best_group_per_row(measurement.values, y_tol=35.0)
        measurement.identifyValueType()

        # Step 4: Post-processing corrections
        changed = False
        if len(measurement.values) >= 1:
            changed |= _try_fix_group(measurement.values[0], "sys")
        if len(measurement.values) >= 2:
            changed |= _try_fix_group(measurement.values[1], "dia")
        if len(measurement.values) >= 3:
            changed |= _try_fix_group(measurement.values[2], "pul")

        if changed:
            measurement.identifyValueType()

        # Step 5: Validate and finalize readings
        sys_raw = measurement.sys()
        dia_raw = measurement.dia()
        pul_raw = measurement.pul()

        sys_val = validate_bp_value(sys_raw, "sys")
        dia_val = validate_bp_value(dia_raw, "dia")
        pul_val = validate_bp_value(pul_raw, "pul")

        avg_conf = round(measurement.averageConfidence(), 4)
        min_conf = round(measurement.lowestConfidence(), 4)
        max_conf = round(measurement.highestConfidence(), 4)

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

        # Step 6: Generate annotated preview if requested
        annotated_b64 = None
        if generate_annotated_image:
            try:
                import base64
                plot_img = res.plot()  # BGR numpy array with drawn boxes
                _, buffer = cv2.imencode(".jpg", plot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not encode annotated preview: {e}")

        total_time_ms = round((time.time() - t_start) * 1000, 2)
        logger.info(f"[{status.upper()}] {msg} (conf={avg_conf:.2f}, time={total_time_ms}ms)")

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
            "detections": raw_detections,
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
