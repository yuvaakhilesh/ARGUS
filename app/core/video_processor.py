import os
import time
import collections
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Generator

from app.core.config import (
    NUM_FRAMES_PER_WINDOW,
    FRAME_INPUT_SIZE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_STRIDE_SECONDS,
    THUMBNAILS_DIR,
    CLIPS_DIR,
    NORMAL_LABEL,
    SEVERITY_MAPPING,
    PEAK_CONFIDENCE_THRESHOLD,
    STEALING_ANALYSIS_ENABLED,
    STEALING_CANDIDATE_THRESHOLD,
    STEALING_CONTEXT_BUFFER_SECONDS,
    STEALING_FRAME_SAMPLE_FPS,
)
from app.core.model import VideoAnomalyDetector
from app.core.stealing_detector import StealingObjectDetector

logger = logging.getLogger("surveillance.video")

class VideoProcessor:
    def __init__(self, detector: Optional[VideoAnomalyDetector] = None):
        self.detector = detector or VideoAnomalyDetector.get_instance()

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Reads metadata from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Re-check actual readable frames if container metadata reported <= 1
        if total_frames <= 1 and cap.isOpened():
            actual_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while cap.grab():
                actual_count += 1
            if actual_count > 0:
                total_frames = actual_count

        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": round(duration, 2),
        }

    def assess_video_quality(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        max_samples: int = 16,
    ) -> Dict[str, Any]:
        """
        Validates video quality and informative content before running heavy inference.
        Detects:
        - Failed decoding or zero usable frames
        - Insufficient temporal length (< 8 frames or < 0.35s)
        - Completely black video (mean/max intensity ~0, zero variance)
        - Blank or near-uniform footage (solid color, uniform gray/white, compression noise only)
        - Static still-frame loops with no visual activity

        Distinguishes real dark surveillance scenes (which have sensor noise, edges,
        gradients, and dynamic range) from uninformative flat frames.
        """
        duration = float(metadata.get("duration", 0.0))
        total_frames = int(metadata.get("total_frames", 0))
        width = int(metadata.get("width", 0))
        height = int(metadata.get("height", 0))

        # 1. Decoding and geometry check
        if width <= 0 or height <= 0:
            return {
                "is_assessable": False,
                "reason": "invalid_geometry",
                "warning": "Video has invalid frame dimensions (0x0). Decoding failed.",
                "metrics": {},
            }

        if total_frames <= 0 or duration <= 0:
            return {
                "is_assessable": False,
                "reason": "zero_frames",
                "warning": "No readable video frames could be decoded from this file.",
                "metrics": {},
            }

        # 2. Minimum temporal length check
        # VideoMAE requires multi-frame temporal context (16 frames per window).
        # Less than 8 frames or < 0.35s is insufficient temporal depth to assess activity.
        if total_frames < 8 or duration < 0.35:
            return {
                "is_assessable": False,
                "reason": "insufficient_duration",
                "warning": f"Video duration ({duration:.2f}s, {total_frames} frames) is too short to capture temporal action patterns (minimum 0.4s / 8 frames required).",
                "metrics": {"total_frames": total_frames, "duration": duration},
            }

        # 3. Sample frames to analyze visual content
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "is_assessable": False,
                "reason": "cannot_open",
                "warning": "Video decoder could not open the media stream.",
                "metrics": {},
            }

        sample_indices = np.linspace(0, max(0, total_frames - 1), min(total_frames, max_samples), dtype=int)
        sampled_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)
        cap.release()

        if not sampled_frames:
            return {
                "is_assessable": False,
                "reason": "zero_usable_frames",
                "warning": "Failed to extract usable frames for visual quality assessment.",
                "metrics": {"usable_frames": 0},
            }

        # Compute spatial, contrast, and temporal statistics
        stds = []
        means = []
        max_vals = []
        dynamic_ranges = []
        lap_vars = []
        temp_diffs = []

        for i, frame in enumerate(sampled_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            stds.append(float(np.std(gray)))
            means.append(float(np.mean(gray)))
            max_vals.append(float(np.max(gray)))
            dynamic_ranges.append(float(np.max(gray) - np.min(gray)))
            lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            if i > 0:
                prev_gray = cv2.cvtColor(sampled_frames[i - 1], cv2.COLOR_BGR2GRAY) if len(sampled_frames[i - 1].shape) == 3 else sampled_frames[i - 1]
                temp_diffs.append(float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))))

        mean_std = float(np.mean(stds))
        mean_intensity = float(np.mean(means))
        max_intensity = float(np.max(max_vals))
        mean_dynamic_range = float(np.mean(dynamic_ranges))
        mean_lap_var = float(np.mean(lap_vars))
        mean_temp_diff = float(np.mean(temp_diffs)) if temp_diffs else 0.0

        metrics = {
            "usable_frames": len(sampled_frames),
            "mean_spatial_std": round(mean_std, 3),
            "mean_intensity": round(mean_intensity, 2),
            "max_intensity": round(max_intensity, 2),
            "mean_dynamic_range": round(mean_dynamic_range, 2),
            "mean_laplacian_var": round(mean_lap_var, 3),
            "mean_temporal_diff": round(mean_temp_diff, 3),
        }

        # 4. Check for completely black / zero-signal video
        if max_intensity < 5.0 and mean_dynamic_range < 5.0:
            return {
                "is_assessable": False,
                "reason": "completely_black_footage",
                "warning": "Video footage is completely black with zero discernible visual content.",
                "metrics": metrics,
            }

        # 5. Check for blank or near-uniform footage (solid color / flat gray / white)
        # Even with compression artifacts, uniform screens have spatial_std < 2.0, dynamic_range < 12, lap_var < 5.0.
        # Genuine dark surveillance has sensor noise with lap_var >= 15.0 or dynamic_range >= 20.0.
        if mean_std < 2.0 and mean_dynamic_range < 12.0 and mean_lap_var < 5.0:
            return {
                "is_assessable": False,
                "reason": "near_uniform_blank_footage",
                "warning": "Video contains blank or near-uniform frames with insufficient visual texture or information.",
                "metrics": metrics,
            }

        # 6. Check for static frozen footage on short clips (< 2.5s)
        # If all frames are virtually identical still frames (temporal_diff < 0.02 and no motion)
        if duration < 2.5 and mean_temp_diff < 0.02 and mean_lap_var < 10.0 and mean_std < 5.0:
            return {
                "is_assessable": False,
                "reason": "static_uninformative_clip",
                "warning": "Video consists of static, unchanging frames with insufficient activity for motion analysis.",
                "metrics": metrics,
            }

        # 7. Check for synthetic screen recordings / computer UI / document / website footage
        # Characteristics: large proportion of flat solid color regions (RGB identical), zero sensor noise,
        # or digital text layout. Real CCTV always has optical sensor noise across the entire field.
        flat_pixel_ratios = []
        for frame in sampled_frames:
            h, w = frame.shape[:2]
            block_stds = []
            for by in range(0, h - 15, 32):
                for bx in range(0, w - 15, 32):
                    block = frame[by:by+32, bx:bx+32]
                    block_stds.append(float(np.std(block)))
            if block_stds:
                flat_pixel_ratios.append(float(np.mean([s < 4.0 for s in block_stds])))
        mean_flat_ratio = float(np.mean(flat_pixel_ratios)) if flat_pixel_ratios else 0.0
        metrics["flat_region_ratio"] = round(mean_flat_ratio, 3)

        if mean_intensity > 50.0 and mean_flat_ratio > 0.40 and mean_temp_diff < 8.0:
            return {
                "is_assessable": False,
                "reason": "unsupported_screen_or_document_recording",
                "warning": "Footage appears to be a digital screen, document, or non-surveillance recording (excessive flat digital regions and lack of optical camera sensor noise).",
                "metrics": metrics,
            }

        return {
            "is_assessable": True,
            "reason": None,
            "warning": None,
            "metrics": metrics,
        }

    def extract_uniform_frames(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, num_frames: int = 16) -> List[np.ndarray]:
        """
        Extracts num_frames uniformly between start_frame and end_frame.
        Fault-tolerant seeking ensures no black frames or codec seek failures.
        """
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        extracted = []
        last_valid = None

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted.append(rgb_frame)
                last_valid = rgb_frame
            else:
                if last_valid is not None:
                    extracted.append(last_valid.copy())
                elif extracted:
                    extracted.append(extracted[-1].copy())
                else:
                    extracted.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Pad if short
        while len(extracted) < num_frames:
            if extracted:
                extracted.append(extracted[-1].copy())
            else:
                extracted.append(np.zeros((224, 224, 3), dtype=np.uint8))

        return extracted

    def stream_windows(
        self,
        video_path: str,
        windows: List[Tuple[float, float]],
        fps: float,
        total_frames: int,
        num_frames: int = NUM_FRAMES_PER_WINDOW,
        cancel_check: Optional[Any] = None,
    ) -> Generator[Tuple[int, Tuple[float, float], List[np.ndarray]], None, None]:
        """
        Incrementally extracts and yields 16-frame uniform windows one by one in a single forward pass.
        Maintains a bounded sliding frame cache in RAM (< 10 MB) regardless of whether the video
        is 10 seconds or 10 hours. Immediately evicts frames that are no longer needed by future windows.
        """
        if not windows:
            return

        window_indices_list = []
        frame_ref_count = collections.Counter()
        for start_t, end_t in windows:
            start_f = max(0, int(start_t * fps))
            end_f = min(int(end_t * fps), max(start_f + 1, total_frames))
            indices = np.linspace(start_f, max(start_f + 1, end_f - 1), num_frames, dtype=int).tolist()
            window_indices_list.append(indices)
            for f in indices:
                frame_ref_count[f] += 1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"stream_windows could not open video: {video_path}")
            return

        frame_cache: Dict[int, np.ndarray] = {}
        current_frame = 0
        last_valid = None
        active_w = 0
        total_w = len(windows)
        max_needed_frame = max(frame_ref_count.keys()) if frame_ref_count else 0

        try:
            while cap.isOpened() and current_frame <= max_needed_frame and active_w < total_w:
                if cancel_check and cancel_check():
                    break

                if frame_ref_count[current_frame] > 0:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # VideoMAE shortest-edge 224 + center crop to (224, 224)
                    # Preserves natural aspect ratio without distortion and produces exact (224, 224, 3) frames
                    h, w = rgb.shape[:2]
                    scale = 224.0 / float(min(h, w))
                    new_w = max(224, int(round(w * scale)))
                    new_h = max(224, int(round(h * scale)))
                    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    crop_x = (new_w - 224) // 2
                    crop_y = (new_h - 224) // 2
                    cropped = resized[crop_y : crop_y + 224, crop_x : crop_x + 224]
                    frame_cache[current_frame] = cropped
                    last_valid = cropped
                else:
                    if not cap.grab():
                        break

                current_frame += 1

                # Yield all windows whose required frames have been decoded up to current_frame
                while active_w < total_w and current_frame > window_indices_list[active_w][-1]:
                    if cancel_check and cancel_check():
                        return
                    fallback = last_valid if last_valid is not None else np.zeros((224, 224, 3), dtype=np.uint8)
                    w_indices = window_indices_list[active_w]
                    w_frames = [frame_cache.get(f, fallback).copy() for f in w_indices]
                    yield active_w, windows[active_w], w_frames

                    # Evict dereferenced frames from cache to keep peak RAM < 10 MB
                    for f in w_indices:
                        frame_ref_count[f] -= 1
                        if frame_ref_count[f] <= 0 and f in frame_cache:
                            del frame_cache[f]
                    active_w += 1

            # If video reached EOF before all windows were yielded (e.g. metadata frame count discrepancy)
            # Only complete the current partially decoded window if at least some frames were decoded up to EOF.
            # Do NOT synthesize phantom windows whose start frame is beyond actual video EOF.
            if active_w < total_w and last_valid is not None and current_frame > window_indices_list[active_w][0]:
                if not (cancel_check and cancel_check()):
                    fallback = last_valid if last_valid is not None else np.zeros((224, 224, 3), dtype=np.uint8)
                    w_indices = window_indices_list[active_w]
                    w_frames = [frame_cache.get(f, fallback).copy() for f in w_indices]
                    yield active_w, windows[active_w], w_frames
                    for f in w_indices:
                        if f in frame_cache:
                            del frame_cache[f]
                    active_w += 1

        finally:
            cap.release()
            frame_cache.clear()

    def extract_all_windows_fast(
        self,
        video_path: str,
        windows: List[tuple],
        fps: float,
        total_frames: int,
        num_frames: int = NUM_FRAMES_PER_WINDOW,
    ) -> List[List[np.ndarray]]:
        """
        Extracts 16-frame uniform windows across the entire video in ONE single forward pass.
        Zero backward seeking, zero duplicate frame decoding.
        Backed by stream_windows with bounded memory.
        """
        all_window_frames = []
        for _, _, w_frames in self.stream_windows(
            video_path=video_path,
            windows=windows,
            fps=fps,
            total_frames=total_frames,
            num_frames=num_frames,
        ):
            all_window_frames.append(w_frames)
        return all_window_frames

    def extract_video_clip(
        self,
        source_video_path: str,
        start_time: float,
        end_time: float,
        output_clip_path: Path,
        overlay_text: Optional[str] = None,
    ) -> bool:
        """
        Extracts a precision-trimmed MP4 clip.
        Strategy 1: FFmpeg re-encode with libx264 (-ss before -i and -t for duration).
        Strategy 2: OpenCV frame-by-frame fallback.
        """
        s_time = max(0.0, float(start_time))
        e_time = max(s_time + 0.5, float(end_time))
        duration = round(e_time - s_time, 3)

        logger.info(f"Extracting clip: {s_time:.2f}s → {e_time:.2f}s (duration={duration:.2f}s) from {source_video_path}")

        ffmpeg_exe = None
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            import shutil
            ffmpeg_exe = shutil.which("ffmpeg")

        if ffmpeg_exe:
            try:
                import subprocess

                # ── Strategy 1: Accurate re-encode (-ss BEFORE -i, -t for duration, libx264) ──
                cmd_encode = [
                    ffmpeg_exe, "-y",
                    "-ss", f"{s_time:.3f}",
                    "-i", str(source_video_path),
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    "-movflags", "+faststart",
                    str(output_clip_path),
                ]
                res = subprocess.run(cmd_encode, capture_output=True, text=True, timeout=300)
                out_path = Path(output_clip_path)
                if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    # Verify output length with OpenCV
                    v_cap = cv2.VideoCapture(str(output_clip_path))
                    actual_duration = 0.0
                    if v_cap.isOpened():
                        c_fps = v_cap.get(cv2.CAP_PROP_FPS) or 25.0
                        c_frames = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        actual_duration = round(c_frames / c_fps, 2) if c_fps > 0 else 0.0
                        v_cap.release()
                    logger.info(
                        f"Re-encode clip OK: {out_path.stat().st_size} bytes | "
                        f"Expected duration: {duration:.2f}s, Verified duration: {actual_duration:.2f}s"
                    )
                    return True
                else:
                    logger.warning(f"FFmpeg re-encode failed (rc={res.returncode}): {res.stderr[-400:]}")

            except Exception as e:
                logger.warning(f"FFmpeg clip extraction error: {e}")

        # ── Strategy 2: OpenCV frame-by-frame extraction (H.264 / browser compatible) ──
        logger.info("Falling back to OpenCV clip extraction")
        try:
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open source video for clip extraction: {source_video_path}")
                return False
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Select browser-compatible H.264 codec
            # On Windows, cv2.CAP_MSMF with H264 provides native hardware/OS H.264 MP4 encoding
            out_writer = None
            codec_candidates = [
                ("MSMF_H264", cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264')),
                ("MSMF_avc1", cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'avc1')),
                ("ANY_avc1", cv2.CAP_ANY, cv2.VideoWriter_fourcc(*'avc1')),
                ("ANY_H264", cv2.CAP_ANY, cv2.VideoWriter_fourcc(*'H264')),
                ("ANY_mp4v", cv2.CAP_ANY, cv2.VideoWriter_fourcc(*'mp4v')),
            ]
            for codec_name, backend, fourcc_val in codec_candidates:
                try:
                    if backend == cv2.CAP_MSMF and os.name != "nt":
                        continue
                    writer = cv2.VideoWriter(str(output_clip_path), backend, fourcc_val, fps, (width, height))
                    if writer.isOpened():
                        out_writer = writer
                        logger.info(f"Initialized OpenCV video writer using {codec_name}")
                        break
                except Exception as writer_err:
                    logger.debug(f"Codec candidate {codec_name} failed: {writer_err}")

            if out_writer is None or not out_writer.isOpened():
                logger.error("Could not initialize any OpenCV VideoWriter codec")
                cap.release()
                return False

            start_f = max(0, int(s_time * fps))
            target_frames = max(1, int(duration * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

            written = 0
            while written < target_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                out_writer.write(frame)
                written += 1

            cap.release()
            out_writer.release()

            out_p = Path(output_clip_path)
            if not out_p.exists() or out_p.stat().st_size < 1000 or written == 0:
                logger.error(f"Clip generation resulted in invalid or empty file (written={written}, size={out_p.stat().st_size if out_p.exists() else 0})")
                if out_p.exists():
                    try:
                        out_p.unlink()
                    except Exception:
                        pass
                return False

            # Verify decodability with OpenCV
            v_check = cv2.VideoCapture(str(output_clip_path))
            v_ok = v_check.isOpened()
            v_frames = v_check.get(cv2.CAP_PROP_FRAME_COUNT) if v_ok else 0
            v_check.release()

            if not v_ok or v_frames <= 0:
                logger.error(f"Extracted clip failed decodability verification: {output_clip_path}")
                try:
                    out_p.unlink()
                except Exception:
                    pass
                return False

            logger.info(f"OpenCV clip extraction verified: {written} frames, size={out_p.stat().st_size} bytes")
            return True
        except Exception as e2:
            logger.error(f"OpenCV fallback clip failed: {e2}", exc_info=True)
            return False

    def analyze_video(
        self,
        video_path: str,
        window_duration: float = DEFAULT_WINDOW_SECONDS,
        stride_duration: float = DEFAULT_STRIDE_SECONDS,
        video_id: str = "video",
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes sliding-window VideoMAE analysis across the entire video.
        Uses single-pass sequential sampling to eliminate repeated seeks.
        """
        if cancel_check and cancel_check():
            return None

        start_time = time.time()

        if progress_callback:
            progress_callback("Validating video quality", 10)

        metadata = self.get_video_metadata(video_path)
        duration = metadata["duration"]
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]

        # Step 0: Assess video quality before running VideoMAE or raising crime alerts
        quality_assessment = self.assess_video_quality(video_path, metadata)
        if not quality_assessment["is_assessable"]:
            logger.warning(
                f"Video quality validation rejected '{video_path}': "
                f"reason={quality_assessment['reason']}, warning='{quality_assessment['warning']}'"
            )
            total_processing_time_s = round(time.time() - start_time, 2)
            warning_msg = quality_assessment["warning"]
            if progress_callback:
                progress_callback("Completed (Unassessable)", 100)

            empty_probs = {cls: 0.0 for cls in self.detector.class_names}
            return {
                "verdict": "Unassessable: Insufficient visual information",
                "is_unassessable": True,
                "quality_warning": warning_msg,
                "top_class": "Unassessable",
                "confidence": 0.0,
                "severity": "NONE",
                "overall_threat": "NONE",
                "summary": (
                    f"Video footage cannot be reliably evaluated: {warning_msg} "
                    f"No behavioral anomalies or crime alerts were raised."
                ),
                "anomaly_percentage": 0.0,
                "max_anomaly_score": 0.0,
                "total_windows_analyzed": 0,
                "incidents": [],
                "clips": [],
                "timeline": [],
                "probabilities_14_class": empty_probs,
                "peak_probabilities_14_class": empty_probs,
                "probabilities_breakdown": {
                    "mean": empty_probs,
                    "peak": empty_probs,
                    "mean_sum": 0.0,
                },
                "metadata": metadata,
                "total_processing_time_s": total_processing_time_s,
                "model_used": "OPear/videomae-large-finetuned-UCF-Crime",
                "device": str(self.detector.device),
                "engine_mode": self.detector.engine_mode,
                "quality_metrics": quality_assessment.get("metrics", {}),
                "responsible_ai_notice": "Video quality validation prevented spurious false alarms on uninformative footage.",
            }

        if duration <= 0:
            raise ValueError("Video duration is 0 seconds or invalid.")

        if progress_callback:
            progress_callback("Extracting frames", 20)

        timeline: List[Dict[str, Any]] = []

        # Generate temporal windows
        current_start = 0.0
        window_idx = 0

        if duration <= window_duration:
            windows = [(0.0, duration)]
        else:
            windows = []
            while current_start + window_duration <= duration:
                windows.append((round(current_start, 2), round(current_start + window_duration, 2)))
                current_start += stride_duration
            if current_start < duration:
                windows.append((round(max(0.0, duration - window_duration), 2), round(duration, 2)))

        if cancel_check and cancel_check():
            return None

        # Bounded-memory streaming inference: evaluate windows incrementally as they are decoded
        total_windows = len(windows)
        for idx, (w_idx, (start_t, end_t), clip_frames) in enumerate(
            self.stream_windows(
                video_path=video_path,
                windows=windows,
                fps=fps,
                total_frames=total_frames,
                num_frames=NUM_FRAMES_PER_WINDOW,
                cancel_check=cancel_check,
            )
        ):
            if cancel_check and cancel_check():
                return None

            if progress_callback:
                pct = 20 + int(((idx + 1) / max(1, total_windows)) * 65)
                progress_callback(f"Analyzing window {idx + 1}/{total_windows} ({start_t:.1f}s)", min(85, pct))

            # Window-level quality check: identify static scenes, graphic outro cards, or frozen uninformative frames
            grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip_frames]
            diffs = [
                float(np.mean(np.abs(grays[k].astype(float) - grays[k - 1].astype(float))))
                for k in range(1, len(grays))
            ]
            win_motion = float(np.mean(diffs)) if diffs else 0.0
            win_max_diff = float(np.max(diffs)) if diffs else 0.0
            bright_frame_count = sum(1 for g in grays if np.mean(g) > 175.0)

            is_rejected_quality = False
            rejection_reason = ""
            if win_motion < 0.2:
                is_rejected_quality = True
                rejection_reason = (
                    f"Rejected by window-quality check: completely frozen scene with negligible motion "
                    f"({win_motion:.2f} < 0.2); suppressed false alarm."
                )
            elif (bright_frame_count >= 2 and win_max_diff > 50.0) or (bright_frame_count >= 1 and win_max_diff > 80.0):
                is_rejected_quality = True
                rejection_reason = (
                    f"Rejected by window-quality check: scene transition into bright graphic outro card "
                    f"({bright_frame_count}/16 bright frames, cut max_diff={win_max_diff:.1f}); suppressed false alarm."
                )
            elif bright_frame_count >= 10 and win_motion < 6.0:
                is_rejected_quality = True
                rejection_reason = (
                    f"Rejected by window-quality check: graphic title/outro screen "
                    f"({bright_frame_count}/16 bright frames, motion={win_motion:.2f} < 6.0); suppressed false alarm."
                )

            if is_rejected_quality:
                pred = {
                    "top_label": "Normal",
                    "final_prediction": "Normal",
                    "confidence": 1.0,
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "severity": "NONE",
                    "decision_reason": rejection_reason,
                    "raw_scores": {cls: (1.0 if cls == "Normal" else 0.0) for cls in self.detector.class_names},
                    "top_3": [{"label": "Normal", "prob": 1.0}],
                    "class_probabilities": {cls: (1.0 if cls == "Normal" else 0.0) for cls in self.detector.class_names},
                    "threshold": self.detector.anomaly_threshold,
                    "is_rejected_quality": True,
                }
            else:
                pred = self.detector.predict_clip(clip_frames)

            timeline_point = {
                "window_index": window_idx,
                "start_time": start_t,
                "end_time": end_t,
                "center_time": round((start_t + end_t) / 2.0, 2),
                "top_label": pred["top_label"],
                "final_prediction": pred.get("final_prediction", pred["top_label"]),
                "confidence": pred["confidence"],
                "anomaly_score": pred["anomaly_score"],
                "is_anomaly": pred["is_anomaly"],
                "severity": pred["severity"],
                "decision_reason": pred.get("decision_reason", ""),
                "raw_scores": pred.get("raw_scores", pred.get("class_probabilities", {})),
                "top_3": pred["top_3"],
                "class_probabilities": pred.get("class_probabilities", {}),
                "raw_logits": pred.get("raw_logits", []),
                "is_rejected_quality": is_rejected_quality,
                "rejection_reason": rejection_reason,
            }
            timeline.append(timeline_point)
            window_idx += 1

        if cancel_check and cancel_check():
            return None

        # DEBUG LOGGING (Step 1 requested by user)
        logger.info("=== TIMELINE WINDOWS DEBUG ===")
        for pt in timeline:
            logger.info(
                f"Window {pt['window_index']:02d}: {pt['start_time']:.2f}s - {pt['end_time']:.2f}s | "
                f"center={pt['center_time']:.2f}s | label={pt['top_label']} | "
                f"conf={pt['confidence']:.4f} | score={pt['anomaly_score']:.4f} | anomaly={pt['is_anomaly']}"
            )

        if progress_callback:
            progress_callback("Generating summary", 85)

        # Aggregate incidents and extract clipped evidence videos
        incidents = self._aggregate_incidents(timeline, fps, video_id, video_path)

        logger.info("=== INCIDENTS FROM AGGREGATE DEBUG ===")
        for inc in incidents:
            logger.info(
                f"Incident {inc['incident_id']}: {inc['crime_type']} | "
                f"start_time={inc['start_time']}s, end_time={inc['end_time']}s | "
                f"peak_timestamp={inc['peak_timestamp']}s | "
                f"clip_start={inc['clip_start']}s, clip_end={inc['clip_end']}s | "
                f"clip_file={inc['clip_filename']}"
            )

        if cancel_check and cancel_check():
            return None

        anomaly_windows = [pt for pt in timeline if pt["is_anomaly"]]
        anomaly_percentage = round((len(anomaly_windows) / len(timeline) * 100) if timeline else 0.0, 1)
        max_anomaly_score = max([pt["anomaly_score"] for pt in timeline], default=0.0)

        critical_incidents = [inc for inc in incidents if inc["severity"] == "CRITICAL"]
        high_incidents = [inc for inc in incidents if inc["severity"] == "HIGH"]

        if critical_incidents:
            overall_threat = "CRITICAL"
        elif high_incidents or len(incidents) > 1:
            overall_threat = "HIGH"
        elif incidents:
            overall_threat = "MEDIUM"
        else:
            overall_threat = "NORMAL"

        # Overall verdict & top class
        has_anomaly = len(incidents) > 0
        verdict = "Anomaly detected" if has_anomaly else "Normal"

        # Step 3: Probability distribution across probe classes: mean across windows and peak value
        detector = self.detector
        class_names = list(detector.class_names)
        total_w_cnt = max(1, len(timeline))
        class_probs_sum = {cls: 0.0 for cls in class_names}
        class_probs_peak = {cls: 0.0 for cls in class_names}
        for pt in timeline:
            probs = pt.get("class_probabilities", {})
            for cls in class_names:
                val = probs.get(cls, 0.0)
                class_probs_sum[cls] += val
                if val > class_probs_peak[cls]:
                    class_probs_peak[cls] = val
        mean_breakdown = {k: round(v / total_w_cnt, 4) for k, v in class_probs_sum.items()}
        peak_breakdown = {k: round(v, 4) for k, v in class_probs_peak.items()}

        logger.info(f"Mean probability breakdown sum: {sum(mean_breakdown.values()):.4f} (~100%)")

        if incidents:
            top_incident = max(incidents, key=lambda x: x["confidence"])
            top_class = top_incident["crime_type"]
            overall_confidence = top_incident["confidence"]
            overall_severity = top_incident["severity"]
        else:
            top_class = "Normal Videos"
            overall_confidence = mean_breakdown.get("Normal", mean_breakdown.get("Normal Videos", round(1.0 - max_anomaly_score, 4)))
            overall_severity = "NONE"

        # Generate human-readable summary
        if incidents:
            if len(incidents) == 1:
                inc = incidents[0]
                summary = (
                    f"Anomaly detected: {inc['crime_type']} flagged from "
                    f"{self._format_time(inc['start_time'])} to {self._format_time(inc['end_time'])} "
                    f"with {int(inc['confidence']*100)}% peak confidence ({inc['severity']} severity). "
                    f"Normal activity observed for the remainder of the footage."
                )
            else:
                types = list({inc['crime_type'] for inc in incidents})
                types_str = ", ".join(types)
                summary = (
                    f"Multiple anomalies detected ({len(incidents)} incidents: {types_str}). "
                    f"Primary event ({top_class}) peaked at {int(overall_confidence*100)}% confidence "
                    f"({overall_severity} severity). Review flagged segments below."
                )
        else:
            summary = (
                "Normal surveillance footage. No behavioral anomalies or security violations "
                "detected across the entire recording timeline."
            )

        # Clean standardized timeline representation
        clean_timeline = [
            {
                "start": pt["start_time"],
                "end": pt["end_time"],
                "class": pt["top_label"],
                "score": pt["anomaly_score"],
                "is_anomaly": pt["is_anomaly"],
                "severity": pt["severity"],
                "confidence": pt["confidence"],
                "raw_logits": pt.get("raw_logits", []),
            }
            for pt in timeline
        ]

        # Standardized clips
        clean_clips = [
            {
                "clip_id": inc["incident_id"],
                "start": inc.get("clip_start", inc["start_time"]),
                "end": inc.get("clip_end", inc["end_time"]),
                "cluster_start": inc["start_time"],
                "cluster_end": inc["end_time"],
                "clip_start": inc.get("clip_start", inc["start_time"]),
                "clip_start_s": inc.get("clip_start", inc["start_time"]),
                "clip_end": inc.get("clip_end", inc["end_time"]),
                "duration": max(0.1, round(inc.get("clip_end", inc["end_time"]) - inc.get("clip_start", inc["start_time"]), 2)),
                "cluster_duration": max(0.1, inc["duration"]),
                "class": inc["crime_type"],
                "confidence": inc["confidence"],
                "severity": inc["severity"],
                "review_priority": inc.get("review_priority", "STANDARD"),
                "source_video_id": inc.get("source_video_id", video_id),
                "thumbnail_url": inc["thumbnail_url"],
                "clip_url": inc["clip_url"],
                "clip_filename": inc.get("clip_filename", ""),
                "clip_is_trimmed": bool(inc.get("clip_is_trimmed", False)),
                "description": inc["description"],
                "stealing_analysis": inc.get("stealing_analysis"),
                "weapon_analysis": inc.get("weapon_analysis"),
            }
            for inc in incidents
        ]

        total_processing_time_s = round(time.time() - start_time, 2)

        if progress_callback:
            progress_callback("Generating summary", 100)

        return {
            "verdict": verdict,
            "top_class": top_class,
            "confidence": round(overall_confidence, 4),
            "severity": overall_severity,
            "summary": summary,
            "timeline": clean_timeline,
            "clips": clean_clips,
            "probabilities_14_class": mean_breakdown,
            "peak_probabilities_14_class": peak_breakdown,
            "probabilities_breakdown": {
                "mean": mean_breakdown,
                "peak": peak_breakdown,
                "mean_sum": round(sum(mean_breakdown.values()), 4),
            },
            # Backward-compatibility keys
            "metadata": metadata,
            "overall_threat": overall_threat,
            "anomaly_percentage": anomaly_percentage,
            "max_anomaly_score": round(max_anomaly_score, 4),
            "incidents_count": len(incidents),
            "incidents": incidents,
            "full_timeline": timeline,
            "total_processing_time_s": total_processing_time_s,
            "model_used": "OPear/videomae-large-finetuned-UCF-Crime",
            "device": str(self.detector.device),
            "engine_mode": self.detector.engine_mode,
            "responsible_ai_notice": "AI detection triage signal for human review; does not constitute legal proof of intent.",
        }

    def _aggregate_incidents(
        self,
        timeline: List[Dict[str, Any]],
        fps: float,
        video_id: str,
        video_path: str,
        low_conf_threshold: float = 0.35,
        low_conf_run: int = 2,
        max_clip_seconds: float = 30.0,
        max_gap_windows: int = 1,
        high_conf_single_window_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Groups sequential anomaly segments into distinct incidents.

        Splitting & Merging rules:
        - Allows a brief normal gap of up to `max_gap_windows` (default 1 window)
          to bridge momentary score dips during sustained anomalous events.
        - Splits if >= (max_gap_windows + 1) consecutive normal windows occur.
        - Within a long anomaly run, split whenever confidence drops below
          `low_conf_threshold` for `low_conf_run` consecutive windows.
        - Multi-window clusters (>=2) form standard incidents.
        - Isolated 1-window anomalies are preserved only if confidence >= high_conf_single_window_threshold (e.g. >= 0.75),
          preventing spurious single-window noise while preserving high-confidence brief incidents.
        - Each resulting cluster is capped to `max_clip_seconds` centred on its peak window.
        """
        incidents = []
        current_cluster: List[Dict[str, Any]] = []
        gap_buffer: List[Dict[str, Any]] = []
        low_run = 0

        def flush_cluster(cluster: List[Dict[str, Any]]):
            peak_conf = max((pt.get("confidence", 0.0) for pt in cluster), default=0.0)
            if len(cluster) >= 2:
                if peak_conf < PEAK_CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"Cluster at {cluster[0]['start_time']:.2f}s - {cluster[-1]['end_time']:.2f}s "
                        f"peaked at {peak_conf:.4f} < {PEAK_CONFIDENCE_THRESHOLD}; discarding as sub-threshold noise."
                    )
                    return
                incident = self._build_incident(
                    cluster, len(incidents) + 1, fps, video_id, video_path,
                    max_clip_seconds=max_clip_seconds,
                )
                incidents.append(incident)
            elif len(cluster) == 1:
                pt_iso = cluster[0]
                if pt_iso.get("confidence", 0.0) >= high_conf_single_window_threshold:
                    logger.info(
                        f"Preserving high-confidence short event at {pt_iso['start_time']:.2f}s - {pt_iso['end_time']:.2f}s "
                        f"({pt_iso['top_label']}, conf={pt_iso['confidence']:.4f} >= {high_conf_single_window_threshold})."
                    )
                    incident = self._build_incident(
                        cluster, len(incidents) + 1, fps, video_id, video_path,
                        max_clip_seconds=max_clip_seconds,
                    )
                    incident["short_event_preserved"] = True
                    incidents.append(incident)
                else:
                    logger.info(
                        f"Isolated single anomaly window at {pt_iso['start_time']:.2f}s - {pt_iso['end_time']:.2f}s "
                        f"({pt_iso['top_label']}, conf={pt_iso['confidence']:.4f}) reported as low-confidence; not creating clip."
                    )

        for pt in timeline:
            if pt["is_anomaly"]:
                # If crime class changed, flush previous cluster as distinct event
                if current_cluster and pt["top_label"] != current_cluster[0]["top_label"]:
                    flush_cluster(current_cluster)
                    current_cluster = []
                    gap_buffer = []
                    low_run = 0

                # If there were buffered gap windows within tolerance of same class, bridge them into cluster
                if gap_buffer and len(gap_buffer) <= max_gap_windows:
                    current_cluster.extend(gap_buffer)
                gap_buffer = []

                # Check for sustained low confidence inside an active cluster
                if current_cluster and pt["confidence"] < low_conf_threshold:
                    low_run += 1
                    if low_run >= low_conf_run:
                        flush_cluster(current_cluster)
                        current_cluster = []
                        low_run = 0
                        continue
                else:
                    low_run = 0
                current_cluster.append(pt)
            else:
                low_run = 0
                if current_cluster:
                    gap_buffer.append(pt)
                    if len(gap_buffer) > max_gap_windows:
                        flush_cluster(current_cluster)
                        current_cluster = []
                        gap_buffer = []

        if current_cluster:
            flush_cluster(current_cluster)

        return incidents

    def _build_incident(
        self,
        cluster: List[Dict[str, Any]],
        incident_num: int,
        fps: float,
        video_id: str,
        video_path: str,
        max_clip_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Creates a consolidated incident record with full evidentiary metadata.
        The evidence clip is capped at `max_clip_seconds` centred on the peak window.
        """
        cluster_start = cluster[0]["start_time"]
        cluster_end = cluster[-1]["end_time"]
        cluster_duration = round(cluster_end - cluster_start, 2)

        peak_point = max(cluster, key=lambda x: x["anomaly_score"])
        peak_time = peak_point["center_time"]
        crime_label = peak_point["top_label"]
        peak_score = peak_point["anomaly_score"]
        confidence = peak_point["confidence"]
        severity = SEVERITY_MAPPING.get(crime_label, "MEDIUM")

        # Derive human review priority from severity and confidence
        if severity == "CRITICAL":
            review_priority = "P1-CRITICAL"
        elif severity == "HIGH":
            review_priority = "P1-HIGH" if confidence >= 0.70 else "P2-ELEVATED"
        elif severity == "MEDIUM":
            review_priority = "P2-ELEVATED" if confidence >= 0.80 else "P3-MEDIUM"
        else:
            review_priority = "P4-LOW"

        # Pre-event and post-event padding for human review context
        pre_pad = 1.0
        post_pad = 1.0
        clip_start_raw = max(0.0, cluster_start - pre_pad)
        clip_end_raw = cluster_end + post_pad

        padded_duration = clip_end_raw - clip_start_raw
        if padded_duration <= max_clip_seconds:
            clip_start = round(clip_start_raw, 2)
            clip_end = round(clip_end_raw, 2)
        else:
            half = max_clip_seconds / 2.0
            clip_start = max(clip_start_raw, round(peak_time - half, 2))
            clip_end = round(clip_start + max_clip_seconds, 2)
            if clip_end > clip_end_raw:
                clip_end = round(clip_end_raw, 2)
                clip_start = max(clip_start_raw, round(clip_end - max_clip_seconds, 2))

        # 1. Save Peak Thumbnail — open a short-lived cap just for the thumbnail frame
        thumbnail_filename = f"{video_id}_incident_{incident_num}.jpg"
        thumbnail_path = THUMBNAILS_DIR / thumbnail_filename
        thumbnail_url = ""
        try:
            cap_thumb = cv2.VideoCapture(video_path)
            if cap_thumb.isOpened():
                peak_frame_num = int(peak_time * fps)
                cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_num)
                ret, frame = cap_thumb.read()
                if ret and frame is not None:
                    thumb = cv2.resize(frame, (480, 270))
                    cv2.imwrite(str(thumbnail_path), thumb)
                    thumbnail_url = f"/static/thumbnails/{thumbnail_filename}"
            cap_thumb.release()
        except Exception as e:
            logger.warning(f"Thumbnail generation failed for incident {incident_num}: {e}")

        # 2. Extract and save actual trimmed Evidence Video Clip (.mp4)
        clip_filename = f"{video_id}_incident_{incident_num}.mp4"
        clip_path = CLIPS_DIR / clip_filename
        clip_success = self.extract_video_clip(
            source_video_path=video_path,
            start_time=clip_start,
            end_time=clip_end,
            output_clip_path=clip_path,
        )
        clip_url = f"/static/clips/{clip_filename}" if clip_success else ""
        clip_is_trimmed = bool(clip_success)

        # 3. Stealing-Specific Object Detection & Tracking (Conditional & Compute-Efficient)
        stealing_analysis = None
        is_stealing_predicted = (crime_label == "Stealing")
        max_stealing_score = 0.0
        for pt in cluster:
            raw = pt.get("raw_logits") or []
            if isinstance(raw, (list, tuple)) and len(raw) == 4:
                max_stealing_score = max(max_stealing_score, float(raw[3]))
            elif isinstance(pt.get("class_probabilities"), dict):
                max_stealing_score = max(max_stealing_score, float(pt["class_probabilities"].get("Stealing", 0.0)))

        is_stealing_candidate = is_stealing_predicted or (
            max_stealing_score >= STEALING_CANDIDATE_THRESHOLD and confidence < 0.70
        )

        if STEALING_ANALYSIS_ENABLED and is_stealing_candidate:
            trigger_reason = (
                f"Stealing predicted as primary class ({int(confidence*100)}%)"
                if is_stealing_predicted
                else f"Stealing candidate score ({max_stealing_score:.4f}) >= threshold ({STEALING_CANDIDATE_THRESHOLD})"
            )
            logger.info(f"Triggering stealing object analysis for incident {incident_num} ({trigger_reason})")
            try:
                stealing_detector = StealingObjectDetector.get_instance()
                stealing_analysis = stealing_detector.analyze_candidate_clip(
                    video_path=video_path,
                    start_time=clip_start,
                    end_time=clip_end,
                    context_buffer_seconds=STEALING_CONTEXT_BUFFER_SECONDS,
                    sample_fps=STEALING_FRAME_SAMPLE_FPS,
                )
                stealing_analysis["candidate_trigger_reason"] = trigger_reason
                stealing_analysis["clip_start_s"] = round(clip_start, 2)
                stealing_analysis["clip_is_trimmed"] = clip_is_trimmed
            except Exception as e:
                logger.error(f"Stealing analysis failed gracefully: {e}")
                stealing_analysis = {
                    "status": "error",
                    "error": str(e),
                    "candidate_trigger_reason": trigger_reason,
                    "clip_start_s": round(clip_start, 2),
                    "clip_is_trimmed": clip_is_trimmed,
                    "uncalibrated_interaction_score": 0.0,
                    "review_recommendation": review_priority,
                    "limitations": f"Object tracker encountered error: {e}. VideoMAE anomaly prediction preserved.",
                }
        else:
            stealing_analysis = {
                "status": "skipped_not_candidate",
                "enabled": STEALING_ANALYSIS_ENABLED,
                "clip_start_s": round(clip_start, 2),
                "clip_is_trimmed": clip_is_trimmed,
                "candidate_trigger_reason": f"Incident class '{crime_label}' did not meet stealing candidate threshold ({STEALING_CANDIDATE_THRESHOLD}).",
                "uncalibrated_interaction_score": 0.0,
                "review_recommendation": "NOT_APPLICABLE",
                "decision_reason": f"Incident classified as {crime_label} (anomaly score {peak_score:.2f}). Stealing-specific object tracking applies only to stealing candidates.",
            }

        # 4. Dedicated Weapon Detection & Generalized Multi-Object Tracking (All Incidents)
        weapon_analysis = None
        try:
            from app.core.tracking_pipeline import GeneralizedTracker
            gen_tracker = GeneralizedTracker.get_instance()
            tracking_res = gen_tracker.analyze_clip(
                video_path=video_path,
                start_time=clip_start,
                end_time=clip_end,
                context_buffer_seconds=1.0,
                sample_fps=10.0,
                run_weapons=True,
            )
            weapon_analysis = tracking_res.get("weapon_analysis", {})
            weapon_analysis["track_summaries"] = tracking_res.get("track_summaries", [])
            weapon_analysis["objects_detected"] = tracking_res.get("objects_detected", [])
            weapon_analysis["inventory"] = tracking_res.get("inventory", {})

            # If confirmed weapon is detected, elevate incident review priority
            if weapon_analysis.get("confirmed_weapons"):
                if review_priority in ("P3-MEDIUM", "P4-LOW", "P2-ELEVATED"):
                    review_priority = "P1-HIGH"

            # If stealing_analysis was skipped because not a candidate, populate track_summaries so UI can inspect them
            if stealing_analysis and stealing_analysis.get("status") == "skipped_not_candidate":
                stealing_analysis["track_summaries"] = tracking_res.get("track_summaries", [])
                stealing_analysis["objects_detected"] = tracking_res.get("objects_detected", [])
        except Exception as e:
            logger.error(f"Generalized tracking / weapon analysis error: {e}", exc_info=True)
            weapon_analysis = {
                "status": "error",
                "error": str(e),
                "cues": [],
                "weapon_count": 0,
                "weapons_detected": [],
                "review_recommendation": "MANUAL_REVIEW_REQUIRED",
                "decision_reason": f"Weapon detector encountered error: {e}",
            }

        return {
            "incident_id": f"INC-{incident_num:03d}",
            "crime_type": crime_label,
            "start_time": cluster_start,
            "end_time": cluster_end,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "duration": cluster_duration,
            "peak_timestamp": peak_time,
            "anomaly_score": round(peak_score, 4),
            "confidence": round(confidence, 4),
            "severity": severity,
            "review_priority": review_priority,
            "source_video_id": video_id,
            "model_version": getattr(self.detector, "engine_mode", "probe_4class") + " (VideoMAE backbone)",
            "thumbnail_url": thumbnail_url,
            "clip_url": clip_url,
            "clip_filename": clip_filename,
            "clip_is_trimmed": clip_is_trimmed,
            "description": (
                f"{crime_label} detected between "
                f"{self._format_time(cluster_start, force_decimal=((cluster_duration < 1.0) or (int(cluster_end) == int(cluster_start) and cluster_end > cluster_start)))} and "
                f"{self._format_time(cluster_end, force_decimal=((cluster_duration < 1.0) or (int(cluster_end) == int(cluster_start) and cluster_end > cluster_start)))} "
                f"with {int(confidence*100)}% peak confidence (Review Priority: {review_priority})."
            ),
            "stealing_analysis": stealing_analysis,
            "weapon_analysis": weapon_analysis,
            "responsible_ai_disclaimer": "AI detection triage signal for human review; does not establish legal proof of criminal intent.",
        }

    @staticmethod
    def _format_time(seconds: float, force_decimal: bool = False) -> str:
        if seconds < 0:
            seconds = 0.0
        mins = int(seconds // 60)
        secs = seconds % 60
        if force_decimal or (0.0 < seconds < 1.0):
            return f"{mins:02d}:{secs:04.1f}"
        rounded_secs = int(round(secs))
        if rounded_secs == 60:
            mins += 1
            rounded_secs = 0
        return f"{mins:02d}:{rounded_secs:02d}"
