"""
Model layer for VideoMAE (OPear/videomae-large-finetuned-UCF-Crime).

Public interface (used by video_processor.py):
    VideoAnomalyDetector.get_instance()
    detector.predict_clip(frames) -> dict
    detector.device, detector.is_loaded, detector.loading_error

Input:  list of HxWx3 uint8 RGB frames (any count; resampled to the model's num_frames)
Output: dict with top_label, confidence, anomaly_score, is_anomaly, severity,
        top_3, class_probabilities

Decision rule (per window):
    p_normal = P("Normal Videos")
    top_anom = highest-probability class other than "Normal Videos"
    p_top    = its probability
    is_anomaly = top_anom is an anomaly class
                 AND p_top >= ANOMALY_THRESHOLD
                 AND p_top > p_normal
    anomaly_score = p_top if is_anomaly else p_normal
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from app.core.config import (
    ANOMALY_THRESHOLD,
    NORMAL_LABEL,
    SEVERITY_MAPPING,
)

logger = logging.getLogger("surveillance.model")

MODEL_ID = "OPear/videomae-large-finetuned-UCF-Crime"

# Checkpoint label → canonical name used by config.py and the UI.
# Any checkpoint label not listed here is kept as-is.
# Fails loudly (via the NORMAL_LABEL guard) if a required label is absent.
LABEL_ALIASES: Dict[str, str] = {
    "Normal_Videos_event": "Normal Videos",
    "RoadAccidents": "Road Accidents",
}


class VideoAnomalyDetector:
    _instance: Optional["VideoAnomalyDetector"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VideoAnomalyDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.is_loaded = False
        self.loading_error: Optional[str] = None
        self.id2label: Dict[int, str] = {}
        self.num_frames: int = 0
        self.load_model()

    # ------------------------------------------------------------------ loading
    def load_model(self) -> None:
        """Load the model eagerly. Failures are stored and re-raised, never hidden."""
        if self.is_loaded:
            return
        logger.info(f"Loading {self.model_id} on {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_id)
            self.model.to(self.device).eval()

            self.id2label = {
                int(k): LABEL_ALIASES.get(v, v)
                for k, v in self.model.config.id2label.items()
            }
            self.num_frames = int(self.model.config.num_frames)

            if NORMAL_LABEL not in self.id2label.values():
                raise RuntimeError(
                    f"NORMAL_LABEL '{NORMAL_LABEL}' not in model labels: "
                    f"{sorted(self.id2label.values())}"
                )

            self.is_loaded = True
            self.loading_error = None
            logger.info(
                f"Model loaded: {len(self.id2label)} classes, "
                f"num_frames={self.num_frames}, "
                f"image_mean={self.processor.image_mean}, "
                f"image_std={self.processor.image_std}"
            )
        except Exception as e:
            import traceback
            self.loading_error = traceback.format_exc()
            self.is_loaded = False
            logger.error(f"Model load failed:\n{self.loading_error}")
            raise

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                f"VideoMAE model is not loaded. Load error:\n{self.loading_error}"
            )

    # ----------------------------------------------------------- preprocessing
    def _resample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Pick exactly num_frames frames, evenly spaced across the window."""
        n = len(frames)
        if n == 0:
            raise ValueError("predict_clip received no frames")
        idx = np.linspace(0, n - 1, self.num_frames).round().astype(int)
        return [frames[i] for i in idx]

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def _class_probabilities(self, frames: List[np.ndarray]) -> Dict[str, float]:
        self._ensure_loaded()
        picked = self._resample_frames(frames)

        # The processor resizes and normalizes using the model's own config.
        inputs = self.processor(picked, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}

    def predict_clip(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        probs = self._class_probabilities(frames)

        p_normal = probs[NORMAL_LABEL]
        anomaly_items = {k: v for k, v in probs.items() if k != NORMAL_LABEL}
        top_anom = max(anomaly_items, key=anomaly_items.get)
        p_top = anomaly_items[top_anom]

        is_anomaly = (
            p_top >= ANOMALY_THRESHOLD
            and p_top > p_normal
        )

        if is_anomaly:
            top_label = top_anom
            confidence = p_top
            anomaly_score = p_top
            severity = SEVERITY_MAPPING.get(top_anom, "MEDIUM")
        else:
            top_label = NORMAL_LABEL
            confidence = p_normal
            anomaly_score = p_normal
            severity = "NONE"

        top_3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

        return {
            "top_label": top_label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": bool(is_anomaly),
            "severity": severity,
            "top_3": [{"label": k, "prob": round(v, 4)} for k, v in top_3],
            "class_probabilities": {k: round(v, 4) for k, v in probs.items()},
        }
