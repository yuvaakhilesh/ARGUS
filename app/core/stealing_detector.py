"""
AEGIS Stealing-Specific Object Detection and Multi-Object Tracking Subsystem.

Provides visual evidence analysis for suspected stealing incidents using:
- Pretrained lightweight YOLOv8n object detector (lazy loaded singleton).
- ByteTrack multi-object tracker for persistent ID tracking within candidate clips.
- Heuristic interaction analysis: person-object proximity, approach/departure,
  object displacement, and object disappearance cues.
- Grounded, uncalibrated triage evidence schema combined with VideoMAE predictions.
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
    YOLO_WEIGHTS_PATH,
    STEALING_ANALYSIS_ENABLED,
    STEALING_CANDIDATE_THRESHOLD,
    STEALING_CONTEXT_BUFFER_SECONDS,
    STEALING_FRAME_SAMPLE_FPS,
    STEALING_DETECTION_CONF_THRESHOLD,
    STEALING_PERSON_CLASS_IDS,
    STEALING_TARGET_CLASSES,
    STEALING_PROXIMITY_THRESHOLD,
    STEALING_DISPLACEMENT_THRESHOLD,
)

logger = logging.getLogger("surveillance.stealing_detector")


class StealingObjectDetector:
    """
    Lazy-loaded singleton for YOLOv8n object detection and ByteTrack tracking.
    Thread-safe and reused across candidate incident inspections.
    """
    _instance: Optional["StealingObjectDetector"] = None
    _lock = threading.Lock()

    def __init__(self, weights_path: Optional[Path] = None):
        self.weights_path = Path(weights_path or YOLO_WEIGHTS_PATH)
        self.model: Any = None
        self.is_loaded: bool = False
        self.loading_error: Optional[str] = None
        self.device: Any = 0 if torch.cuda.is_available() else "cpu"
        self.license_info: str = "AGPL-3.0 (Ultralytics YOLOv8)"
        self.supported_classes: Dict[int, str] = {}
        self.target_classes: Dict[int, str] = dict(STEALING_TARGET_CLASSES)
        self.person_classes: List[int] = list(STEALING_PERSON_CLASS_IDS)

    @classmethod
    def get_instance(cls, weights_path: Optional[Path] = None) -> "StealingObjectDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(weights_path)
            return cls._instance

    def load_model(self) -> bool:
        """Lazily load YOLOv8 model weights."""
        if self.is_loaded and self.model is not None:
            return True

        with self._lock:
            if self.is_loaded and self.model is not None:
                return True

            try:
                from ultralytics import YOLO

                # Fallback path search if configured path does not exist
                target_weights = self.weights_path
                if not target_weights.exists():
                    root_weights = Path("yolov8n.pt")
                    if root_weights.exists():
                        target_weights = root_weights
                    else:
                        target_weights = Path("yolov8n.pt")

                logger.info(f"Loading lightweight object detector from: {target_weights}")
                self.model = YOLO(str(target_weights))
                if torch.cuda.is_available():
                    try:
                        self.model.to(self.device)
                    except Exception as e_dev:
                        logger.warning(f"Failed to move general detector to {self.device}: {e_dev}")
                self.supported_classes = getattr(self.model, "names", {})
                self.is_loaded = True
                self.loading_error = None
                logger.info(f"YOLO detector successfully initialized ({len(self.supported_classes)} classes available).")
                return True
            except Exception as e:
                self.is_loaded = False
                self.loading_error = str(e)
                logger.error(f"Failed to load YOLO object detector: {e}", exc_info=True)
                return False

    def reset_tracker(self):
        """
        Explicitly resets tracker state between independent videos or clips.
        Ensures track IDs are never leaked or reused across unrelated videos.
        """
        if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                for t in self.model.predictor.trackers:
                    if hasattr(t, "reset"):
                        try:
                            t.reset()
                        except Exception:
                            pass
            self.model.predictor.trackers = []

    def detect_and_track_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = STEALING_DETECTION_CONF_THRESHOLD,
        filter_class_ids: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        Executes multi-object tracking across a sequential series of frames from a single clip.
        When filter_class_ids is None, all 80 supported COCO classes are tracked.
        Returns:
            - frame_detections: list of detections per frame
            - track_summaries: dictionary mapping track_id -> summary metrics
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Object detector is unavailable: {self.loading_error}")

        if not frames:
            return [], {}

        # Reset tracker before beginning a new video clip session
        self.reset_tracker()

        # If filter_class_ids is explicitly provided, filter by it.
        # Otherwise (None), do NOT restrict: pass classes=None to track all 80 COCO classes.
        target_classes_arg = filter_class_ids if filter_class_ids is not None else None

        all_detections: List[Dict[str, Any]] = []
        track_data: Dict[int, Dict[str, Any]] = {}

        h, w = frames[0].shape[:2]
        diag = math.sqrt(h * h + w * w) if (h > 0 and w > 0) else 1.0

        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Track with ByteTrack
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=tracker_type,
                    classes=target_classes_arg,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"Tracker error on frame {frame_idx}: {e}")
                continue

            if not results or len(results) == 0:
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            frame_records = []
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                track_id = int(b.id[0].item()) if (b.id is not None) else None
                class_name = self.supported_classes.get(cls_id, f"class_{cls_id}")

                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                det = {
                    "frame_index": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "model_source": "coco_yolov8n",
                    "is_person": cls_id in self.person_classes,
                    "is_target_object": cls_id in self.target_classes,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "normalized_bbox": [
                        round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)
                    ],
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "normalized_centroid": [round(cx / w, 4), round(cy / h, 4)],
                    "confidence": round(conf, 4),
                }
                frame_records.append(det)

                # Aggregate track lifecycle metrics
                if track_id is not None:
                    if track_id not in track_data:
                        track_data[track_id] = {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "is_person": cls_id in self.person_classes,
                            "is_target_object": cls_id in self.target_classes,
                            "first_seen_s": timestamp,
                            "last_seen_s": timestamp,
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "detection_count": 0,
                            "max_confidence": conf,
                            "centroids": [],
                            "positions": [],
                            "raw_bboxes_norm": [],
                            "raw_confidences": [],
                        }
                    rec = track_data[track_id]
                    rec["last_seen_s"] = timestamp
                    rec["last_frame"] = frame_idx
                    rec["detection_count"] += 1
                    rec["max_confidence"] = max(rec["max_confidence"], conf)
                    rec["centroids"].append([cx / w, cy / h])
                    rec["positions"].append({"t": round(timestamp, 3), "bbox": det["bbox"]})

                    norm_bbox = [
                        round(max(0.0, min(1.0, x1 / w)), 4),
                        round(max(0.0, min(1.0, y1 / h)), 4),
                        round(max(0.0, min(1.0, x2 / w)), 4),
                        round(max(0.0, min(1.0, y2 / h)), 4),
                    ]
                    rec["raw_bboxes_norm"].append({"t": round(timestamp, 3), "bbox_norm": norm_bbox})
                    rec["raw_confidences"].append({"t": round(timestamp, 3), "conf": round(conf, 4)})

            all_detections.extend(frame_records)

        # Compute summary metrics for each track
        summaries: Dict[int, Dict[str, Any]] = {}
        for tid, tinfo in track_data.items():
            duration = round(tinfo["last_seen_s"] - tinfo["first_seen_s"], 2)
            centroids = tinfo["centroids"]
            displacement = 0.0
            if len(centroids) >= 2:
                dx = centroids[-1][0] - centroids[0][0]
                dy = centroids[-1][1] - centroids[0][1]
                displacement = math.sqrt(dx * dx + dy * dy)

            # Detect gaps in tracking
            expected_frames = tinfo["last_frame"] - tinfo["first_frame"] + 1
            gap_count = max(0, expected_frames - tinfo["detection_count"])

            # Cap bbox_trajectory and confidence_history at 300 points by even subsampling
            raw_boxes = tinfo.get("raw_bboxes_norm", [])
            raw_confs = tinfo.get("raw_confidences", [])
            if len(raw_boxes) > 300:
                indices = np.linspace(0, len(raw_boxes) - 1, 300, dtype=int)
                bbox_traj = [raw_boxes[i] for i in indices]
                conf_hist = [raw_confs[i] for i in indices]
            else:
                bbox_traj = list(raw_boxes)
                conf_hist = list(raw_confs)

            avg_conf = (
                sum(c["conf"] for c in conf_hist) / len(conf_hist)
                if conf_hist else tinfo["max_confidence"]
            )
            summaries[tid] = {
                "track_id": tid,
                "class_id": tinfo.get("class_id"),
                "class_name": tinfo["class_name"],
                "model_source": "coco_yolov8n",
                "is_person": tinfo["is_person"],
                "is_target_object": tinfo["is_target_object"],
                "first_seen_s": round(tinfo["first_seen_s"], 2),
                "last_seen_s": round(tinfo["last_seen_s"], 2),
                "duration_s": duration,
                "detection_count": tinfo["detection_count"],
                "gap_count": gap_count,
                "max_confidence": round(tinfo["max_confidence"], 4),
                "avg_confidence": round(avg_conf, 4),
                "total_displacement": round(displacement, 4),
                "is_moving": bool(displacement > STEALING_DISPLACEMENT_THRESHOLD),
                "bbox_trajectory": bbox_traj,
                "confidence_history": conf_hist,
            }

        # Clear predictor tracker references
        self.reset_tracker()
        return all_detections, summaries

    def analyze_interactions(
        self,
        detections: List[Dict[str, Any]],
        track_summaries: Dict[int, Dict[str, Any]],
        proximity_threshold: float = STEALING_PROXIMITY_THRESHOLD,
        displacement_threshold: float = STEALING_DISPLACEMENT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Evaluates grounded visual interaction cues between tracked persons and objects:
        1. Proximity: Person and object within proximity threshold.
        2. Approach / Movement: Person trajectory closing distance to object.
        3. Object Displacement: Object coordinates significantly shifting while near person.
        4. Object Disappearance: Object consistently observed near person then ceases detection.
        5. Contact Duration: Sustained temporal interaction window.

        Outputs an explainable evidence summary and uncalibrated triage score.
        """
        interaction_cues: List[Dict[str, Any]] = []

        # Partition tracks
        person_tracks = [t for t in track_summaries.values() if t["is_person"]]
        object_tracks = [t for t in track_summaries.values() if t["is_target_object"]]

        if not person_tracks and not object_tracks:
            return {
                "status": "no_relevant_entities",
                "cues": [],
                "uncalibrated_interaction_score": 0.0,
                "decision_reason": "No persons or target theft-associated objects (bags, luggage, phones) detected.",
                "review_recommendation": "INSUFFICIENT_VISUAL_EVIDENCE",
                "limitations": "Pretrained detector did not detect relevant object categories in candidate window.",
            }

        if not person_tracks:
            return {
                "status": "objects_only",
                "cues": [],
                "uncalibrated_interaction_score": 0.10,
                "decision_reason": f"Detected objects ({', '.join({t['class_name'] for t in object_tracks})}) but no persons present in frame.",
                "review_recommendation": "LOW_PRIORITY",
                "limitations": "Absence of detected person precludes person-object theft interaction.",
            }

        if not object_tracks:
            return {
                "status": "persons_only",
                "cues": [],
                "uncalibrated_interaction_score": 0.15,
                "decision_reason": f"Detected {len(person_tracks)} person track(s), but no target personal belongings identified.",
                "review_recommendation": "STANDARD_REVIEW",
                "limitations": "Stolen items may be obscured, out of frame, or unsupported by general object classes.",
            }

        # Index detections by frame
        frames_map: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            frames_map.setdefault(d["frame_index"], []).append(d)

        # Track pairwise distances over time
        proximity_events = []
        displacement_events = []
        disappearance_events = []

        for obj in object_tracks:
            obj_id = obj["track_id"]
            obj_class = obj["class_name"]

            # Check for displacement
            if obj["total_displacement"] >= displacement_threshold:
                displacement_events.append({
                    "object_id": obj_id,
                    "object_class": obj_class,
                    "displacement": obj["total_displacement"],
                    "start_s": obj["first_seen_s"],
                    "end_s": obj["last_seen_s"],
                })

            # Check proximity with each person track across common frames
            for person in person_tracks:
                p_id = person["track_id"]
                distances = []
                common_timestamps = []

                for frame_idx, dets in frames_map.items():
                    p_det = next((d for d in dets if d["track_id"] == p_id), None)
                    o_det = next((d for d in dets if d["track_id"] == obj_id), None)
                    if p_det and o_det:
                        p_cx, p_cy = p_det["normalized_centroid"]
                        o_cx, o_cy = o_det["normalized_centroid"]
                        dist = math.sqrt((p_cx - o_cx) ** 2 + (p_cy - o_cy) ** 2)
                        distances.append(dist)
                        common_timestamps.append(p_det["timestamp"])

                if distances:
                    min_dist = min(distances)
                    if min_dist <= proximity_threshold:
                        contact_frames = sum(1 for d in distances if d <= proximity_threshold)
                        start_t = min(common_timestamps)
                        end_t = max(common_timestamps)
                        contact_duration = round(end_t - start_t, 2)

                        proximity_events.append({
                            "person_id": p_id,
                            "object_id": obj_id,
                            "object_class": obj_class,
                            "min_distance": round(min_dist, 3),
                            "contact_duration_s": contact_duration,
                            "contact_frames": contact_frames,
                            "start_s": start_t,
                            "end_s": end_t,
                        })

                        # Check approach trend
                        if len(distances) >= 3:
                            initial_dist = np.mean(distances[:2])
                            final_dist = np.mean(distances[-2:])
                            if initial_dist - final_dist > 0.05:
                                interaction_cues.append({
                                    "cue_type": "person_approaching_object",
                                    "description": f"Person (Track #{p_id}) converged toward {obj_class} (Track #{obj_id}).",
                                    "start_s": start_t,
                                    "end_s": end_t,
                                    "intensity": round(float(initial_dist - final_dist), 3),
                                })

            # Check disappearance cue:
            # If object was tracked with >=3 detections, ended before the clip ended,
            # and a person continued to be tracked nearby
            if obj["detection_count"] >= 3:
                max_clip_time = max((d["timestamp"] for d in detections), default=0.0)
                if obj["last_seen_s"] < (max_clip_time - 0.8):
                    # Check if any person remained present after object disappeared
                    active_person_after = any(
                        p["last_seen_s"] >= obj["last_seen_s"] + 0.5 for p in person_tracks
                    )
                    if active_person_after:
                        disappearance_events.append({
                            "object_id": obj_id,
                            "object_class": obj_class,
                            "disappearance_timestamp": obj["last_seen_s"],
                        })

        # Register proximity cues
        for pe in proximity_events:
            interaction_cues.append({
                "cue_type": "person_object_proximity",
                "description": (
                    f"Person (Track #{pe['person_id']}) within close reach of {pe['object_class']} "
                    f"(Track #{pe['object_id']}) for {pe['contact_duration_s']}s (min dist: {pe['min_distance']})."
                ),
                "start_s": pe["start_s"],
                "end_s": pe["end_s"],
                "duration_s": pe["contact_duration_s"],
            })

        # Register displacement cues
        for de in displacement_events:
            interaction_cues.append({
                "cue_type": "object_displacement",
                "description": (
                    f"{de['object_class']} (Track #{de['object_id']}) displaced across scene "
                    f"(normalized movement: {de['displacement']:.2f})."
                ),
                "start_s": de["start_s"],
                "end_s": de["end_s"],
                "displacement": de["displacement"],
            })

        # Register disappearance cues
        for dis in disappearance_events:
            interaction_cues.append({
                "cue_type": "object_disappearance",
                "description": (
                    f"{dis['object_class']} (Track #{dis['object_id']}) ceased detection after proximity, "
                    f"while person track persisted in frame."
                ),
                "timestamp_s": dis["disappearance_timestamp"],
            })

        # Calculate uncalibrated heuristic triage score
        # Base score starts from presence of person + target object
        score = 0.20
        reasons = []

        if proximity_events:
            score += 0.30
            reasons.append("sustained person-object proximity")

        if displacement_events:
            score += 0.25
            reasons.append("object position displacement")

        if disappearance_events:
            score += 0.20
            reasons.append("object disappearance from scene")

        uncalibrated_score = min(1.0, round(score, 2))

        # Determine review recommendation
        if uncalibrated_score >= 0.70:
            review_recommendation = "PRIORITY_HUMAN_REVIEW"
        elif uncalibrated_score >= 0.40:
            review_recommendation = "STANDARD_REVIEW"
        else:
            review_recommendation = "LOW_PRIORITY"

        if reasons:
            decision_reason = (
                f"Visual interaction cues detected: {', '.join(reasons)}. "
                f"Person and target object tracked concurrently."
            )
        else:
            decision_reason = (
                f"Persons ({len(person_tracks)}) and objects ({len(object_tracks)}) observed concurrently, "
                f"but no direct proximity or displacement interaction detected."
            )

        return {
            "status": "completed",
            "cues": interaction_cues,
            "uncalibrated_interaction_score": uncalibrated_score,
            "decision_reason": decision_reason,
            "review_recommendation": review_recommendation,
            "limitations": (
                "Uncalibrated visual cue score for human review prioritization. "
                "Does not establish ownership, consent, or legal intent to steal."
            ),
        }

    def analyze_candidate_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        context_buffer_seconds: float = STEALING_CONTEXT_BUFFER_SECONDS,
        sample_fps: float = STEALING_FRAME_SAMPLE_FPS,
    ) -> Dict[str, Any]:
        """
        Extracts candidate interval with context buffer, samples frames efficiently,
        tracks objects with ByteTrack, and returns complete grounded stealing analysis.
        """
        t0 = time.perf_counter()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "error": f"Unable to open video: {video_path}",
                "clip_start_s": round(start_time, 2),
                "uncalibrated_interaction_score": 0.0,
                "review_recommendation": "INSUFFICIENT_VISUAL_EVIDENCE",
                "limitations": "Video could not be opened for object inspection.",
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Apply context buffer
        buffered_start = max(0.0, start_time - context_buffer_seconds)
        buffered_end = min(duration, end_time + context_buffer_seconds)
        if buffered_end <= buffered_start:
            buffered_end = min(duration, buffered_start + 1.0)

        # Sample frames at sample_fps
        step_s = 1.0 / max(1.0, sample_fps)
        sample_times = np.arange(buffered_start, buffered_end, step_s).tolist()
        if not sample_times:
            sample_times = [buffered_start]

        frames: List[np.ndarray] = []
        timestamps: List[float] = []

        for st in sample_times:
            frame_num = min(total_frames - 1, int(st * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Keep frame in BGR for YOLO (YOLO natively converts or expects uint8 BGR/RGB)
                frames.append(frame)
                timestamps.append(st)

        cap.release()

        if not frames:
            return {
                "status": "no_frames_extracted",
                "clip_start_s": round(start_time, 2),
                "uncalibrated_interaction_score": 0.0,
                "review_recommendation": "INSUFFICIENT_VISUAL_EVIDENCE",
                "limitations": "Zero frames retrieved in target interval.",
            }

        try:
            detections, track_summaries = self.detect_and_track_frames(frames, timestamps)
            interaction = self.analyze_interactions(detections, track_summaries)
            proc_time = round(time.perf_counter() - t0, 3)

            # Unique object classes identified
            objects_detected = sorted(list({t["class_name"] for t in track_summaries.values()}))

            return {
                "status": interaction.get("status", "completed"),
                "enabled": True,
                "tracker_type": "ByteTrack",
                "clip_start_s": round(start_time, 2),
                "analyzed_interval": {
                    "start_time": round(buffered_start, 2),
                    "end_time": round(buffered_end, 2),
                    "context_buffer_seconds": context_buffer_seconds,
                    "frames_analyzed": len(frames),
                    "sample_fps": sample_fps,
                },
                "objects_detected": objects_detected,
                "track_summaries": list(track_summaries.values()),
                "interaction_cues": interaction.get("cues", []),
                "uncalibrated_interaction_score": interaction.get("uncalibrated_interaction_score", 0.0),
                "decision_reason": interaction.get("decision_reason", "Analysis completed."),
                "review_recommendation": interaction.get("review_recommendation", "STANDARD_REVIEW"),
                "processing_time_s": proc_time,
                "limitations": interaction.get("limitations", ""),
            }

        except Exception as e:
            logger.error(f"Error during stealing analysis: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "clip_start_s": round(start_time, 2),
                "uncalibrated_interaction_score": 0.0,
                "review_recommendation": "MANUAL_REVIEW_REQUIRED",
                "limitations": f"Detector encountered error: {e}. VideoMAE anomaly prediction preserved.",
                "processing_time_s": round(time.perf_counter() - t0, 3),
            }
