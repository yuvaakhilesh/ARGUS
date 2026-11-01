"""
ARGUS Phase 2 — Face Detection, Quality Assessment & ArcFace Embedding.

Technology selected after evaluation:
  • InsightFace buffalo_sc pack
      - Face detector: SCRFD-500M (onnxruntime, ~6 ms/frame on RTX 4050)
        Architecture: multi-scale CNN with adaptive receptive field
        Weights source: https://github.com/deepinsight/insightface (MIT License)
      - Face recognizer: MobileNet-based ArcFace (mbf), ~3 ms/emb
        Embedding dimension: 512 float32 values
        Trained on MS1M-ArcFace; cosine similarity metric
        Weights source: InsightFace model zoo (MIT License)

  Why InsightFace over alternatives:
    • Pure ONNX inference — no PyTorch conflict with existing VideoMAE/YOLO stack
    • buffalo_sc downloads ~320 MB once, cached locally
    • GPU acceleration through onnxruntime-gpu (CUDA 12.8 compatible)
    • Competitive accuracy on LFW-style benchmarks (MBF: 99.1% @ LFW)
    • MIT-licensed model weights; no redistribution restriction
    • Well-maintained, production-tested on CCTV-quality inputs

  Limitations:
    • ArcFace is trained on posed/frontal-dominant data.
      Detection rate drops significantly at >45° yaw or extreme occlusion.
    • CCTV-resolution faces below 40 px short-side produce unreliable embeddings.
    • Matching threshold (cosine ≥ 0.50 → "possible match") is NOT validated
      on this specific ARGUS deployment's surveillance conditions.
      All matches MUST be treated as advisory and require human review.

Separation of concerns:
  FaceProcessor orchestrates detection → quality → embedding.
  Matching is performed separately by WatchlistMatcher.
  Results are NEVER treated as conclusive identity verification.
"""

import io
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.watchlist_config import (
    FACE_DETECT_MIN_SIZE,
    FACE_DETECT_THRESHOLD,
    FACE_QUALITY_BLUR_THRESHOLD,
    FACE_QUALITY_MIN_SIZE_PX,
    INSIGHTFACE_MODEL_DIR,
    INSIGHTFACE_MODEL_NAME,
)

logger = logging.getLogger("argus.face_processor")


class FaceDetectionResult:
    """Container for a single detected face."""

    __slots__ = (
        "bbox",         # [x1, y1, x2, y2] in pixels (source frame)
        "landmark_5",  # 5-point landmark array or None
        "det_score",   # SCRFD detection confidence (0-1)
        "embedding",   # numpy float32 array (512,) or None
        "quality",     # 'good' | 'low' | 'blurred' | 'tiny'
        "quality_score",  # Laplacian variance (float)
        "face_crop",   # numpy H×W×3 crop (BGR)
    )

    def __init__(self, bbox, det_score, landmark_5=None):
        self.bbox = bbox
        self.det_score = float(det_score)
        self.landmark_5 = landmark_5
        self.embedding: Optional[np.ndarray] = None
        self.quality: str = "unknown"
        self.quality_score: float = 0.0
        self.face_crop: Optional[np.ndarray] = None


class FaceProcessor:
    """
    Thread-safe InsightFace pipeline: detect → quality → embed.

    Singleton pattern mirrors the existing ARGUS detector conventions.
    """

    _instance: Optional["FaceProcessor"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._app = None          # insightface.app.FaceAnalysis
        self._model_lock = threading.Lock()
        self._loaded = False
        self._loading_error: Optional[str] = None
        self.device_id: int = 0   # CUDA device index; -1 for CPU

    @classmethod
    def get_instance(cls) -> "FaceProcessor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loading_error(self) -> Optional[str]:
        return self._loading_error

    def load_model(self, gpu_id: int = 0) -> bool:
        """
        Load InsightFace buffalo_sc on first call.
        Downloads models (~320 MB) to INSIGHTFACE_MODEL_DIR on first run.
        """
        with self._model_lock:
            if self._loaded:
                return True
            try:
                import insightface  # type: ignore
                from insightface.app import FaceAnalysis  # type: ignore

                logger.info(
                    "Loading InsightFace %s. Models will be downloaded to %s on first run.",
                    INSIGHTFACE_MODEL_NAME,
                    INSIGHTFACE_MODEL_DIR,
                )
                self._app = FaceAnalysis(
                    name=INSIGHTFACE_MODEL_NAME,
                    root=str(INSIGHTFACE_MODEL_DIR),
                    providers=[
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                )
                # ctx_id=0 → GPU 0 via CUDA EP; ctx_id=-1 → CPU
                self._app.prepare(ctx_id=gpu_id, det_size=(640, 640))
                self.device_id = gpu_id
                self._loaded = True
                self._loading_error = None
                logger.info("InsightFace %s loaded successfully.", INSIGHTFACE_MODEL_NAME)
                return True
            except Exception as exc:
                self._loading_error = str(exc)
                logger.error("Failed to load InsightFace: %s", exc, exc_info=True)
                return False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect_faces(
        self, frame_bgr: np.ndarray
    ) -> List[FaceDetectionResult]:
        """
        Run SCRFD detection on a BGR frame.
        Returns list of FaceDetectionResult with bbox and det_score.
        Embeddings are NOT computed here.
        """
        if not self._loaded:
            return []
        try:
            faces = self._app.get(frame_bgr)
        except Exception as exc:
            logger.warning("Face detection error: %s", exc)
            return []

        results = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < FACE_DETECT_THRESHOLD:
                continue
            bbox = [int(v) for v in face.bbox.tolist()]
            lm = getattr(face, "kps", None)
            lm = lm.tolist() if lm is not None else None
            fdr = FaceDetectionResult(bbox=bbox, det_score=score, landmark_5=lm)

            # Embedding already extracted by InsightFace app.get()
            emb = getattr(face, "embedding", None)
            if emb is not None:
                fdr.embedding = emb.astype(np.float32)

            results.append(fdr)

        return results

    def assess_quality(
        self,
        frame_bgr: np.ndarray,
        face_result: FaceDetectionResult,
    ) -> FaceDetectionResult:
        """
        Assess face crop quality: blur, size.
        Sets face_result.quality and .quality_score in-place.
        """
        x1, y1, x2, y2 = face_result.bbox
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        fw = x2 - x1
        fh = y2 - y1

        if fw < FACE_DETECT_MIN_SIZE or fh < FACE_DETECT_MIN_SIZE:
            face_result.quality = "tiny"
            face_result.quality_score = 0.0
            face_result.face_crop = None
            return face_result

        crop = frame_bgr[y1:y2, x1:x2]
        face_result.face_crop = crop.copy()

        # Laplacian variance: blur metric
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face_result.quality_score = lap_var

        short_side = min(fw, fh)

        if short_side < FACE_QUALITY_MIN_SIZE_PX:
            face_result.quality = "tiny"
        elif lap_var < FACE_QUALITY_BLUR_THRESHOLD:
            face_result.quality = "blurred"
        else:
            face_result.quality = "good"

        return face_result

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> List[FaceDetectionResult]:
        """
        Full pipeline for a single frame:
        1. Detect faces
        2. Assess quality
        (Embeddings already extracted by InsightFace app.get())

        Returns list of FaceDetectionResult with all fields populated.
        """
        faces = self.detect_faces(frame_bgr)
        for face in faces:
            self.assess_quality(frame_bgr, face)
        return faces

    def process_image_bytes(
        self,
        image_bytes: bytes,
    ) -> List[FaceDetectionResult]:
        """
        Convenience wrapper: decode image bytes → process frame.
        Used for enrollment reference image validation.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes — unsupported format or corrupt data.")
        return self.process_frame(img)

    def crop_to_jpeg_bytes(
        self,
        face_result: FaceDetectionResult,
        quality: int = 85,
    ) -> Optional[bytes]:
        """Encode face_crop to JPEG bytes for storage or review display."""
        if face_result.face_crop is None:
            return None
        ok, buf = cv2.imencode(".jpg", face_result.face_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return bytes(buf)

