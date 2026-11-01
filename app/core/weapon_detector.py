"""
ARGUS Weapon Detection Subsystem.

Dedicated deep-learning detector for bladed weapons and firearms:
- Pretrained lightweight YOLOv8 weapon model (Pistol, Rifle, Knife).
- Independent of general COCO object detector to prevent misattribution.
- Bounded memory inference and GPU acceleration on NVIDIA RTX 4050.
- Responsible AI safeguards: observations designated as uncalibrated triage indicators
  requiring human verification.
"""

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.core.config import (
    BASE_DIR,
)

logger = logging.getLogger("argus.weapon_detector")

DEFAULT_WEAPON_WEIGHTS_PATH = BASE_DIR / "app" / "core" / "weights" / "yolov8_weapon.pt"
DEFAULT_WEAPON_CONFIDENCE_THRESHOLD = 0.28


class WeaponDetector:
    """
    Singleton detector for dedicated weapon classification (Knife, Pistol, Rifle).
    Maintains clean separation from general object detectors.
    """
    _instance: Optional["WeaponDetector"] = None
    _lock = threading.Lock()

    def __init__(self, weights_path: Optional[Path] = None):
        self.weights_path = Path(weights_path or DEFAULT_WEAPON_WEIGHTS_PATH)
        self.model: Any = None
        self.is_loaded: bool = False
        self.loading_error: Optional[str] = None
        self.device: Any = 0 if torch.cuda.is_available() else "cpu"
        self.device_name: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.license_info: str = "AGPL-3.0 (Ultralytics YOLOv8 / WeaponShield-AI)"
        self.supported_classes: Dict[int, str] = {0: "Pistol", 1: "Rifle", 2: "Knife"}
        self.confidence_threshold: float = DEFAULT_WEAPON_CONFIDENCE_THRESHOLD

    @classmethod
    def get_instance(cls, weights_path: Optional[Path] = None) -> "WeaponDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(weights_path)
            return cls._instance

    def load_model(self) -> bool:
        """Loads YOLO weapon detection model weights."""
        if self.is_loaded and self.model is not None:
            return True

        with self._lock:
            if self.is_loaded and self.model is not None:
                return True

            try:
                from ultralytics import YOLO

                target_weights = self.weights_path
                if not target_weights.exists():
                    # Fallback path check
                    root_weights = BASE_DIR / "yolov8_weapon.pt"
                    if root_weights.exists():
                        target_weights = root_weights
                    else:
                        raise FileNotFoundError(f"Weapon weights not found at {self.weights_path}")

                logger.info(f"Loading dedicated weapon detector from: {target_weights} on {self.device}")
                self.model = YOLO(str(target_weights))
                if torch.cuda.is_available():
                    try:
                        self.model.to(self.device)
                    except Exception as e_dev:
                        logger.warning(f"Failed to move weapon model to {self.device}: {e_dev}")
                # Validate model classes
                names = getattr(self.model, "names", {})
                if isinstance(names, dict) and names:
                    self.supported_classes = {int(k): str(v) for k, v in names.items()}
                self.is_loaded = True
                self.loading_error = None
                logger.info(f"Weapon detector loaded successfully. Classes: {self.supported_classes}")
                return True
            except Exception as e:
                self.is_loaded = False
                self.loading_error = str(e)
                logger.error(f"Failed to load weapon detector: {e}", exc_info=True)
                return False

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """
        Runs weapon detection on a single RGB or BGR frame.
        Returns list of structured detections.
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Weapon detector unavailable: {self.loading_error}")

        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        try:
            results = self.model(
                frame,
                conf=conf,
                device=self.device,
                imgsz=imgsz,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Weapon inference error: {e}")
            return []

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            cls_id = int(b.cls[0].item())
            c = float(b.conf[0].item())
            class_name = self.supported_classes.get(cls_id, f"weapon_{cls_id}")

            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            det = {
                "class_id": cls_id,
                "class_name": class_name,
                "is_weapon": True,
                "weapon_type": "firearm" if class_name.lower() in ["pistol", "rifle", "gun"] else "bladed",
                "confidence": round(c, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "normalized_bbox": [
                    round(max(0.0, min(1.0, x1 / w)), 4),
                    round(max(0.0, min(1.0, y1 / h)), 4),
                    round(max(0.0, min(1.0, x2 / w)), 4),
                    round(max(0.0, min(1.0, y2 / h)), 4),
                ],
                "centroid": [round(cx, 1), round(cy, 1)],
                "normalized_centroid": [round(cx / w, 4), round(cy / h, 4)],
            }
            detections.append(det)

        return detections

    def detect_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs detection across a chronological sequence of frames.
        Attaches source timestamps and frame indices to each detection.
        """
        all_detections = []
        for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_dets = self.detect_frame(frame, conf_threshold=conf_threshold)
            for d in frame_dets:
                d["frame_index"] = idx
                d["timestamp"] = round(timestamp, 3)
                all_detections.append(d)
        return all_detections

