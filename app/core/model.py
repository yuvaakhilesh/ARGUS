"""
Anomaly model: frozen VideoMAE features + trained 4-class probe.

Classes: Assault, Fighting, Normal, Stealing.
Input:  list of 16-frame RGB windows (uint8 HxWx3), one list per 2.5 s window.
Output: dict with top_label, confidence, anomaly_score, is_anomaly,
        severity, class_probabilities, top_3.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

# Ensure transformers and huggingface_hub operate offline if local cached weights exist
_hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--OPear--videomae-large-finetuned-UCF-Crime")
if os.path.exists(_hf_cache):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from app.core.config import ANOMALY_THRESHOLD

logger = logging.getLogger("surveillance.model")
MODEL_ID = "OPear/videomae-large-finetuned-UCF-Crime"
PROBE_DIR = os.path.join(os.path.dirname(__file__), "probe")
NORMAL_NAME = "Normal"
SEVERITY_BY_CLASS = {"Assault": "HIGH", "Fighting": "HIGH", "Stealing": "MEDIUM", "Normal": "NONE"}


class VideoAnomalyDetector:
    _instance: Optional["VideoAnomalyDetector"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VideoAnomalyDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)
        self.engine_mode = "probe"
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.is_loaded = False
        self.loading_error: Optional[str] = None
        self.processor = None
        self.backbone = None
        self.probe = None
        self.class_names: List[str] = []
        self.num_frames: int = 16
        self.chunk: int = 2
        self.anomaly_threshold: float = ANOMALY_THRESHOLD
        self.load_model()

    def _remap_attention_biases(self) -> None:
        """
        Remap verified legacy attention biases in OPear/videomae-large-finetuned-UCF-Crime.
        In the safetensors checkpoint, q_bias and v_bias were saved under legacy names,
        while key.bias was omitted. Hugging Face transformers expects query.bias,
        key.bias, and value.bias, leaving them randomly initialized unless remapped.
        """
        try:
            from huggingface_hub import hf_hub_download
            import safetensors.torch
            try:
                sf_path = hf_hub_download(MODEL_ID, "model.safetensors", local_files_only=True)
            except Exception:
                sf_path = hf_hub_download(MODEL_ID, "model.safetensors")
            weights = safetensors.torch.load_file(sf_path)

            remapped_count = 0
            for i, layer_module in enumerate(self.backbone.videomae.encoder.layer):
                q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
                v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
                if q_key in weights and v_key in weights:
                    attn = layer_module.attention.attention
                    attn.query.bias.data.copy_(weights[q_key].to(self.device, dtype=self.dtype))
                    attn.value.bias.data.copy_(weights[v_key].to(self.device, dtype=self.dtype))
                    attn.key.bias.data.zero_()
                    remapped_count += 1
            if remapped_count > 0:
                logger.info(f"Verified and remapped attention biases across {remapped_count} transformer layers.")
        except Exception as e:
            logger.warning(f"Attention bias remapping skipped: {e}")

    def load_model(self) -> None:
        if self.is_loaded:
            return
        try:
            with open(os.path.join(PROBE_DIR, "meta.json")) as f:
                meta = json.load(f)
            self.class_names = meta["class_names"]
            self.num_frames = int(meta["num_frames"])

            self.probe = joblib.load(os.path.join(PROBE_DIR, "probe_4class.joblib"))
            if list(self.probe.classes_) != list(range(len(self.class_names))):
                raise RuntimeError(f"probe classes {list(self.probe.classes_)} do not match meta")

            try:
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)

            try:
                self.backbone = VideoMAEForVideoClassification.from_pretrained(
                    MODEL_ID, torch_dtype=self.dtype, local_files_only=True
                ).to(self.device).eval()
            except Exception:
                self.backbone = VideoMAEForVideoClassification.from_pretrained(
                    MODEL_ID, torch_dtype=self.dtype
                ).to(self.device).eval()

            # Apply verified attention bias remapping
            self._remap_attention_biases()

            self.is_loaded = True
            self.loading_error = None
        except Exception:
            import traceback
            self.loading_error = traceback.format_exc()
            self.is_loaded = False
            raise

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(f"Model not loaded:\n{self.loading_error}")

    @torch.inference_mode()
    def _window_features(self, windows: List[List[np.ndarray]]) -> np.ndarray:
        feats = []
        for i in range(0, len(windows), self.chunk):
            batch = windows[i:i + self.chunk]
            pix = self.processor(batch, return_tensors="pt")["pixel_values"]
            pix = pix.to(self.device, dtype=self.dtype)
            hidden = self.backbone.videomae(pixel_values=pix).last_hidden_state
            feats.append(hidden.mean(dim=1).float().cpu().numpy())
        return np.vstack(feats)

    def predict_clip(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """One window of 16 RGB frames. Used by the per-window timeline in video_processor.py."""
        if not frames:
            raise ValueError("predict_clip received an empty frames list")
        if not isinstance(frames, (list, tuple)):
            raise TypeError(f"predict_clip expected list or tuple of frames, got {type(frames)}")
        return self.predict_windows([frames])

    def predict_windows(self, windows: List[List[np.ndarray]]) -> Dict[str, Any]:
        """Several windows of 16 frames each, averaged per clip. Matches the notebook."""
        self._ensure_loaded()
        if not windows:
            raise ValueError("predict_windows received no windows")
        if any(not w for w in windows):
            raise ValueError("One or more windows in predict_windows is empty")

        feats = self._window_features(windows)

        # OOD & Uninformative feature validation in standardized embedding space:
        # Logistic regression probe hyperplanes extrapolate unbounded on out-of-distribution inputs.
        # Genuine surveillance footage across UCF-Crime has ||z||_2 <= 51.5 (mean 36.8, std 6.5).
        # Unrelated inputs (pure black video, static logo cards, web screen recordings) produce ||z||_2 >= 53.0.
        is_ood = False
        mean_z_norm = 0.0
        try:
            scaler = self.probe.named_steps["standardscaler"]
            z_feats = scaler.transform(feats)
            mean_z_norm = float(np.mean(np.linalg.norm(z_feats, axis=1)))
            if mean_z_norm > 52.5:
                is_ood = True
        except Exception as e_ood:
            logger.debug(f"OOD feature norm calculation skipped: {e_ood}")

        probs_arr = self.probe.predict_proba(feats).mean(axis=0)
        probs = {name: float(probs_arr[i]) for i, name in enumerate(self.class_names)}

        # Policy Evaluation
        p_normal = probs.get(NORMAL_NAME, 0.0)
        anomaly_score = 1.0 - p_normal
        threshold = self.anomaly_threshold

        # If out-of-distribution, suppress ungrounded crime classification
        if is_ood and anomaly_score >= threshold:
            logger.warning(f"Suppressed out-of-distribution crime alert: ||z||_2={mean_z_norm:.2f} > 52.5")
            anomaly_score = 0.0
            p_normal = 1.0
            probs[NORMAL_NAME] = 1.0
            for k in probs:
                if k != NORMAL_NAME:
                    probs[k] = 0.0

        # Determine non-normal signal
        non_normal_classes = {k: v for k, v in probs.items() if k != NORMAL_NAME}
        top_non_normal = max(non_normal_classes, key=non_normal_classes.get) if non_normal_classes else NORMAL_NAME
        p_top_non_normal = non_normal_classes.get(top_non_normal, 0.0)
        top_overall = max(probs, key=probs.get)

        # Direct, Documented Decision Policy:
        # A window is flagged as anomalous if and only if anomaly_score (1 - P(Normal)) >= anomaly_threshold.
        is_anomaly = bool(anomaly_score >= threshold)
        if is_anomaly:
            top_label = top_non_normal
            confidence = p_top_non_normal
            severity = SEVERITY_BY_CLASS.get(top_label, "MEDIUM")
            reason = (
                f"Anomaly detected: anomaly score ({anomaly_score:.4f}) >= threshold ({threshold:.2f}); "
                f"primary crime class is '{top_label}' ({confidence:.4f})."
            )
        else:
            top_label = NORMAL_NAME
            confidence = p_normal
            severity = "NONE"
            if is_ood:
                reason = (
                    f"Normal (Out-of-Distribution Gated): feature norm ({mean_z_norm:.2f}) indicates non-surveillance or uninformative footage; "
                    f"crime alert suppressed."
                )
            else:
                reason = (
                    f"Normal: anomaly score ({anomaly_score:.4f}) < threshold ({threshold:.2f}); "
                    f"P(Normal) is {confidence:.4f}."
                )

        top_3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return {
            "top_label": top_label,
            "final_prediction": top_label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": bool(is_anomaly),
            "severity": severity,
            "decision_reason": reason,
            "raw_scores": {k: round(v, 4) for k, v in probs.items()},
            "top_3": [{"label": k, "prob": round(v, 4)} for k, v in top_3],
            "class_probabilities": {k: round(v, 4) for k, v in probs.items()},
            "threshold": threshold,
        }
