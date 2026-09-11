"""Optic disc inference, geometry and prior-based calibration, without rendering."""

import math
from pathlib import Path

import os

import cv2
import numpy as np



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
    """Validate a pixel-space box with center coordinates."""
    box = np.asarray(box_xywh, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xywh must contain four finite values")
    if np.any(box[2:] <= 0):
        raise ValueError("Box width and height must be positive")
    return tuple(map(float, box))



# ---- Measurements and approximations from boxes information ---- #
def ellipse_from_box(box_xywh):
    """Build an axis-aligned ellipse from a bounding box.

    Returns:
        dict:
            center_px (tuple[float, float]): Center (x, y), in pixels.
            semi_axes_px (tuple[float, float]): Horizontal and vertical radii,
                in pixels; these are half the box width and height.
            angle_deg (float): Rotation angle, always 0.0.
    """
    cx, cy, width, height = _box(box_xywh)
    return {"center_px": (cx, cy), "semi_axes_px": (width / 2, height / 2),
            "angle_deg": 0.0}


def compute_disc_measurements(box_xywh, image_shape, disc_height_prior_mm=1.92):
    """Estimate scale assuming the prior disc height and isotropic pixels.
    Returns:
        dict:
            width_px (float): Bounding-box width in pixels.
            height_px (float): Bounding-box height in pixels.
            vh_ratio (float): Height divided by width; dimensionless.
            mm_per_px (float): Estimated millimeters per pixel.
            um_per_px (float): Estimated micrometers per pixel.
            image_width_mm (float): Estimated physical image width.
            image_height_mm (float): Estimated physical image height.
            disc_height_prior_mm (float): Assumed physical disc height.
    """
    _, _, width, height = _box(box_xywh)
    _positive(disc_height_prior_mm, "disc_height_prior_mm")
    image_h, image_w = image_shape[:2]
    _positive(image_h, "image height")
    _positive(image_w, "image width")
    pitch = disc_height_prior_mm / height
    return {"width_px": width, "height_px": height, "vh_ratio": height / width,
            "mm_per_px": pitch, "um_per_px": pitch * 1000,
            "image_width_mm": image_w * pitch, "image_height_mm": image_h * pitch,
            "disc_height_prior_mm": float(disc_height_prior_mm)}



# ---- Helper function to build result data ---- #
def _extract_detections(result):
    """Extract boxes and confidence scores from YOLO.
    Returns:
        list[dict]: One record per detection, with:
            confidence (float): YOLO confidence score in [0, 1].
            box_xywh (tuple[float, float, float, float]):
                (center_x, center_y, width, height), in original-image pixels.
     Returns an empty list when no boxes are available.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xywh.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()


    return [
        {
            "confidence": float(score),
            "box_xywh": tuple(map(float, box)),
        }
        for i, (box, score) in enumerate(zip(boxes, scores))
    ]




# ----- main prediction function ---- #
def predict_optic_disc(model, image_path, disc_height_prior_mm=1.92,
                       prediction_confidence=0.25, calibration_confidence=0.7):
    """Infer once on the original image and return geometry and calibration.

    Returns:
        dict:
            image_path (str): Input image path.
            image_shape (tuple[int, int]): Original image (height, width).
            status (str): One of:
                "no_detection": No detection is available.
                "below_calibration_confidence": Geometry is available,
                    but physical measurements are not computed.
                "calibrated": Prior-based physical measurements are available.
            detections (list[dict]): Records described in _extract_detections.
            selected_index (int | None): Index into detections of the
                highest-confidence detection; None if no detection exists.
            ellipse (dict | None): Geometry described in ellipse_from_box;
                None if no detection exists.
            measurements (dict | None): Values described in
                compute_disc_measurements; None unless calibration is accepted.
            disc_height_prior_mm (float): Assumed physical disc height.
            prediction_confidence (float): Minimum confidence used by YOLO.
            calibration_confidence (float): Minimum detection confidence
                required to compute physical measurements.

    Coordinates use original-image pixels. Physical measurements assume
    isotropic pixels. "calibrated" does not imply verified physical accuracy.

    Raises:
        ValueError: Invalid parameters, unreadable image or invalid selected box.
    """

    #Verifications on parameters
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


    #predictions 
    results = model.predict(source=image, conf=prediction_confidence, verbose=False)
    detections = _extract_detections(results[0]) if results else []

    #retreiving best detection if multiple
    index = max(
        range(len(detections)),
        key=lambda i: detections[i]["confidence"],
        default=None,
    )

    # Building output
    output = {"image_path": str(image_path), "image_shape": tuple(image.shape[:2]),
              "status": "no_detection", "detections": detections,
              "selected_index": index, "ellipse": None, "measurements": None,
              "disc_height_prior_mm": float(disc_height_prior_mm),
              "prediction_confidence": prediction_confidence,
              "calibration_confidence": calibration_confidence}

    #fail detection
    if index is None:
        return output

    #Information and calculs on detection 
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
    """
        Process prediction on all the temporal dataset
        temporal dataset generated by dataset_videos notebook

        Return:
            np.ndarray: Object array with shape (number_of_videos, nb_frames)
            each element of the array is the result of the predict_optic_disk function
    """

    root = Path(dir_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")

    
    videos_dir = sorted(
        (path for path in root.iterdir() if path.is_dir()), 
        key=lambda path: path.name,
    )

    files_names =  [f"{i:04d}.png" for i in range(nb_frames)]

    predictions = np.empty((len(videos_dir), nb_frames), dtype=object)

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


    
from ultralytics import YOLO

model = YOLO("optic_disk.pt")
preds = predict_optic_disc_temporal(model, "video_dataset", nb_frames=8)

for video_index, frames in enumerate(preds):
    print(f"\nVideo {video_index} — {len(frames)} frames")
    for frame_index, result in enumerate(frames):
        print(
            f"  [{video_index}, {frame_index}] "
            f"{result['image_path']} | {result['status']}"
        )