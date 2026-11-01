"""
ARGUS Phase 2 — Video Face Analysis Worker.

Processes an uploaded video file for face-watchlist analysis.
Designed to be called from a background thread (same pattern as run_analysis_worker).

Key design decisions:
  • Samples at FACE_SAMPLE_FPS (default 2 fps) to bound processing time and VRAM.
  • Caps at FACE_MAX_FRAMES (default 300) regardless of video length.
  • Deduplicates observations within FACE_DEDUP_SECONDS to avoid spamming
    the review queue for a face visible over several seconds.
  • Never retains a complete video's face crops in memory — crops are written
    to the face-crop directory and the path is stored; the in-memory crop is
    released immediately.
  • Does NOT modify the VideoMAE or weapon-detection results.
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.watchlist_config import (
    FACE_DEDUP_SECONDS,
    FACE_MAX_FRAMES,
    FACE_SAMPLE_FPS,
    WATCHLIST_FACE_CROPS_DIR,
)
from app.core.face_processor import FaceProcessor
from app.core.watchlist_matcher import batch_match_video

logger = logging.getLogger("argus.face_analysis")


def analyze_video_faces(
    video_path: str,
    job_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Main entry point for face-watchlist analysis on an uploaded video.

    Returns:
        {
          "status": "completed" | "no_faces_detected" | "watchlist_empty" |
                    "face_module_unavailable" | "error",
          "frames_sampled": int,
          "faces_detected": int,
          "observations_saved": int,
          "possible_matches": int,
          "obs_ids": [str, ...],
          "advisory": str,
          "processing_time_s": float,
        }
    """
    t_start = time.time()

    def _progress(stage: str, pct: int):
        if progress_callback:
            try:
                progress_callback(stage, pct)
            except Exception:
                pass

    def _cancelled() -> bool:
        return cancel_check() if cancel_check else False

    # ── 1. Ensure face model is loaded ─────────────────────────────────────
    processor = FaceProcessor.get_instance()
    if not processor.is_loaded:
        ok = processor.load_model(gpu_id=0)
        if not ok:
            # Try CPU fallback
            ok = processor.load_model(gpu_id=-1)
        if not ok:
            return {
                "status": "face_module_unavailable",
                "error": processor.loading_error,
                "frames_sampled": 0,
                "faces_detected": 0,
                "observations_saved": 0,
                "possible_matches": 0,
                "obs_ids": [],
                "advisory": "Face processing module unavailable.",
                "processing_time_s": round(time.time() - t_start, 2),
            }

    _progress("Opening video for face analysis", 5)

    # ── 2. Open video ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _err("Cannot open video file", t_start)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(1, round(src_fps / FACE_SAMPLE_FPS))
    max_frames = min(FACE_MAX_FRAMES, total_frames if total_frames > 0 else FACE_MAX_FRAMES)

    _progress("Sampling video frames for face detection", 10)

    # ── 3. Frame sampling loop ─────────────────────────────────────────────
    face_observations: List[Dict] = []
    frames_sampled = 0
    faces_detected = 0
    frame_idx = 0

    # Track last-observed timestamp per "face group" for deduplication
    # Using a simple minimum-distance approach without a full tracker
    last_obs_timestamps: List[float] = []

    video_source = Path(video_path).name

    while True:
        if _cancelled():
            cap.release()
            return _err("Cancelled", t_start, status="cancelled")

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        frames_sampled += 1
        source_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Progress reporting (10–90%)
        if max_frames > 0:
            pct = 10 + int(80 * frames_sampled / max_frames)
            _progress(f"Face analysis — frame {frames_sampled}/{max_frames}", min(pct, 88))

        # Detect and quality-assess faces
        faces = processor.process_frame(frame)

        for face in faces:
            if face.embedding is None:
                continue

            faces_detected += 1

            # Deduplication: skip if a face observation was recently logged
            # within FACE_DEDUP_SECONDS
            too_close = any(
                abs(source_ts - t) < FACE_DEDUP_SECONDS
                for t in last_obs_timestamps[-20:]  # check last 20 obs
            )
            if too_close:
                continue

            last_obs_timestamps.append(source_ts)

            # Save face crop to disk (retention-bounded)
            crop_path = _save_face_crop(
                face.face_crop, video_source, frame_idx, job_id
            )

            face_observations.append({
                "source_timestamp": source_ts,
                "embedding": face.embedding,
                "quality": face.quality,
                "quality_score": face.quality_score,
                "face_crop_path": crop_path,
            })

        if frames_sampled >= max_frames:
            break

    cap.release()

    if faces_detected == 0:
        return {
            "status": "no_faces_detected",
            "frames_sampled": frames_sampled,
            "faces_detected": 0,
            "observations_saved": 0,
            "possible_matches": 0,
            "obs_ids": [],
            "advisory": (
                "No usable faces detected in the selected video. "
                "Face-watchlist analysis produced no observations."
            ),
            "processing_time_s": round(time.time() - t_start, 2),
        }

    _progress("Matching observations against watchlist", 90)

    # ── 4. Match against watchlist ─────────────────────────────────────────
    obs_ids = batch_match_video(
        video_source=video_source,
        job_id=job_id,
        face_observations=face_observations,
    )

    # Summarise outcomes (do not count embeddings; count saved observations)
    from app.core.watchlist_db import list_observations
    saved_obs = list_observations(video_source=video_source)
    possible_count = sum(
        1 for o in saved_obs
        if o["obs_id"] in obs_ids and o["match_result"] == "possible_match"
    )

    # Release embedding arrays from memory
    for obs in face_observations:
        obs.pop("embedding", None)

    _progress("Face analysis complete", 100)

    return {
        "status": "completed",
        "frames_sampled": frames_sampled,
        "faces_detected": faces_detected,
        "observations_saved": len(obs_ids),
        "possible_matches": possible_count,
        "obs_ids": obs_ids,
        "advisory": (
            "AI-generated possible matches. Identity has not been independently established. "
            "Human verification is required for all flagged observations."
        ),
        "processing_time_s": round(time.time() - t_start, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_face_crop(
    crop: Optional[np.ndarray],
    video_source: str,
    frame_idx: int,
    job_id: Optional[str],
) -> Optional[str]:
    """Save face crop JPEG to WATCHLIST_FACE_CROPS_DIR. Returns path or None."""
    if crop is None:
        return None
    try:
        stem = f"{video_source}_{job_id or 'nojob'}_{frame_idx}"
        out_path = WATCHLIST_FACE_CROPS_DIR / f"{stem}.jpg"
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out_path.write_bytes(bytes(buf))
            return str(out_path)
    except Exception as exc:
        logger.debug("Could not save face crop: %s", exc)
    return None


def _err(msg: str, t_start: float, status: str = "error") -> Dict:
    return {
        "status": status,
        "error": msg,
        "frames_sampled": 0,
        "faces_detected": 0,
        "observations_saved": 0,
        "possible_matches": 0,
        "obs_ids": [],
        "advisory": "Face analysis did not complete.",
        "processing_time_s": round(time.time() - t_start, 2),
    }

