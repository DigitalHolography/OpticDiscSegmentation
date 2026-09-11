"""Optic disc inference, geometry and prior-based calibration, without rendering."""

import math
from pathlib import Path


import cv2
import numpy as np

# ----- Structure of different returned types ----- #
DETECTION_DTYPE = np.dtype([("confidence", np.float64), ("box_xywh", np.float64, (4,))])
ELLIPSE_DTYPE = np.dtype([
    ("center_px", np.float64, (2,)), ("semi_axes_px", np.float64, (2,)),
    ("angle_deg", np.float64),
])
MEASUREMENTS_DTYPE = np.dtype([
    (name, np.float64) for name in (
        "width_px", "height_px", "vh_ratio", "mm_per_px", "um_per_px",
        "image_width_mm", "image_height_mm", "disc_height_prior_mm",
    )
])
PREDICTION_DTYPE = np.dtype([
    ("image_path", object), ("image_shape", np.int64, (2,)),
    ("status", "U32"), ("detections", object), ("selected_index", np.int64),
    ("ellipse", ELLIPSE_DTYPE), ("measurements", MEASUREMENTS_DTYPE),
    ("disc_height_prior_mm", np.float64), ("prediction_confidence", np.float64),
    ("calibration_confidence", np.float64),
])




# ----- Value validation functions ----- #
def _positive(value, name):
    """Validate a finite positive scalar."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _confidence(value, name):
    """Validate a confidence threshold."""

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _box(box_xywh):
    """Return a finite (4,) float array: center_x, center_y, width, height."""

    box = np.asarray(box_xywh, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xywh must contain four finite values")
    if np.any(box[2:] <= 0):
        raise ValueError("Box width and height must be positive")
    return box



# ----- Measurements and approximations from boxes information ----- #
def ellipse_from_box(box_xywh):
    """Return a shape-() ELLIPSE_DTYPE array without contour fitting.

    Fields: center_px (x, y), semi_axes_px (horizontal, vertical radii),
    and angle_deg (always zero). Coordinates and radii are in pixels."""

    cx, cy, width, height = _box(box_xywh)

    ellipse = np.zeros((), dtype=ELLIPSE_DTYPE)
    ellipse["center_px"] = (cx, cy)
    ellipse["semi_axes_px"] = (width / 2, height / 2)
    ellipse["angle_deg"] = 0.0

    return ellipse


def compute_disc_measurements(box_xywh, image_shape, disc_height_prior_mm=1.92):
    """Return a shape-() MEASUREMENTS_DTYPE array assuming isotropic pixels.

    Fields: width_px, height_px, vh_ratio (height/width), mm_per_px,
    um_per_px, image_width_mm, image_height_mm, disc_height_prior_mm.
    Physical measurements are estimates based on the disc-height prior."""

    _, _, width, height = _box(box_xywh)
    _positive(disc_height_prior_mm, "disc_height_prior_mm")
    image_h, image_w = image_shape[:2]
    _positive(image_h, "image height")
    _positive(image_w, "image width")
    pitch = disc_height_prior_mm / height

    measurements = np.zeros((), dtype=MEASUREMENTS_DTYPE)
    measurements["width_px"] = width
    measurements["height_px"] = height
    measurements["vh_ratio"] = height / width
    measurements["mm_per_px"] = pitch
    measurements["um_per_px"] = pitch * 1000
    measurements["image_width_mm"] = image_w * pitch
    measurements["image_height_mm"] = image_h * pitch
    measurements["disc_height_prior_mm"] = disc_height_prior_mm

    return measurements



# ----- Helper function to build result data ----- #
def _extract_detections(result):
    """Return an (N,) DETECTION_DTYPE array; N is zero when no boxes exist.

    Fields: confidence in [0, 1] and box_xywh containing
    (center_x, center_y, width, height) in original-image pixels."""

    if result.boxes is None or len(result.boxes) == 0:
        return np.empty(0, dtype=DETECTION_DTYPE)

    boxes = result.boxes.xywh.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()


    detections = np.empty(len(scores), dtype=DETECTION_DTYPE)
    detections["confidence"] = scores
    detections["box_xywh"] = boxes
    return detections




# ----- main prediction function ----- #
def predict_optic_disc(model, image_path, disc_height_prior_mm=1.92,
                       prediction_confidence=0.25, calibration_confidence=0.7):
    """Infer once and return a shape-() PREDICTION_DTYPE array.

    Fields:
        image_path: Input path string stored in an object field.
        image_shape: Integer array (height, width).
        status: "no_detection", "below_calibration_confidence", or "calibrated".
        detections: Variable-length DETECTION_DTYPE array in an object field.
            Retrieve it from this shape-() result with output["detections"][()].
        selected_index: Highest-confidence detection index, or -1 if absent.
        ellipse: ELLIPSE_DTYPE fields; all NaN if no detection exists.
        measurements: MEASUREMENTS_DTYPE fields; all NaN unless calibrated.
        disc_height_prior_mm: Assumed physical disc height.
        prediction_confidence: Minimum confidence used by YOLO.
        calibration_confidence: Minimum detection confidence for calibration.

    Coordinates use original-image pixels. "calibrated" means a prior-based
    estimate is available, not verified physical accuracy. Invalid parameters,
    unreadable images or invalid boxes raise ValueError. Model errors propagate."""

    # Parameters verification
    _positive(disc_height_prior_mm, "disc_height_prior_mm")
    _confidence(prediction_confidence, "prediction_confidence")
    _confidence(calibration_confidence, "calibration_confidence")

    if calibration_confidence < prediction_confidence:
        raise ValueError("Calibration threshold must be >= prediction threshold")

    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")


    # run detection
    results = model.predict(source=image, conf=prediction_confidence, verbose=False)
    detections = _extract_detections(results[0]) if results else np.empty(0, dtype=DETECTION_DTYPE)

    # Select the best detection if several
    index = int(np.argmax(detections["confidence"])) if len(detections) else -1

    # Building output
    output = np.zeros((), dtype=PREDICTION_DTYPE)
    output["image_path"] = str(image_path)
    output["image_shape"] = image.shape[:2]
    output["status"] = "no_detection"
    output["detections"][()] = detections
    output["selected_index"] = index
    output["disc_height_prior_mm"] = disc_height_prior_mm
    output["prediction_confidence"] = prediction_confidence
    output["calibration_confidence"] = calibration_confidence

    # Missing numerical data are NaN 
    for field in ELLIPSE_DTYPE.names:
        output["ellipse"][field] = np.nan
    for field in MEASUREMENTS_DTYPE.names:
        output["measurements"][field] = np.nan

    if index == -1:
        return output

    # Compute geometry and accepted physical measurements
    selected = detections[index]
    output["ellipse"] = ellipse_from_box(selected["box_xywh"])
    output["status"] = "below_calibration_confidence"
    if selected["confidence"] >= calibration_confidence:
        output["measurements"] = compute_disc_measurements(
            selected["box_xywh"], image.shape, disc_height_prior_mm)
        output["status"] = "calibrated"

    return output



# ---- Apply prediction on temoral data ---- #
def predict_optic_disc_temporal(model, dir_path, disc_height_prior_mm=1.92,
                       prediction_confidence=0.25, calibration_confidence=0.7,
                       nb_frames = 8):
    """Return a (number_of_videos, nb_frames) PREDICTION_DTYPE array.

    Rows follow sorted video-directory names; columns follow 0000.png, etc.
    Each record has the fields documented in predict_optic_disc.
    Field access is vectorized: results["measurements"]["mm_per_px"].
    Missing files and prediction errors propagate."""

    root = Path(dir_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")


    videos_dir = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )

    files_names =  [f"{i:04d}.png" for i in range(nb_frames)]

    predictions = np.empty((len(videos_dir), nb_frames), dtype=PREDICTION_DTYPE)

    for vid_idx, currdir in enumerate(videos_dir):
        for frame_idx, currfile in enumerate(files_names):
            predictions[vid_idx, frame_idx] = predict_optic_disc(
                model = model,
                image_path=currdir / currfile,
                disc_height_prior_mm=disc_height_prior_mm,
                prediction_confidence=prediction_confidence,
                calibration_confidence=calibration_confidence,
            )

    return predictions




