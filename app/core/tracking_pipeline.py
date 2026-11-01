"""
ARGUS Generalized Multi-Object Tracking & Threat Association Pipeline.

Features:
- Multi-class tracking using ByteTrack with persistent Kalman state.
- Supports all 80 COCO classes from YOLOv8n + 3 tactical weapon classes from WeaponShield.
- Enforces strict tracker lifecycle: persist across consecutive frames of the same video,
  cleanly reset between independent videos to prevent ID leakage.
- Preserves raw frame coordinates, normalized coordinates, and absolute source timestamps.
- Duplicate detection suppression and model fusion between general and weapon detectors.
- Targeted object scanning and comprehensive object search across analyzed video.
- Object inventory compilation and responsible AI person-weapon proximity evaluation.
- Never fabricates detections, IDs, or threat confirmations.
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
    STEALING_PERSON_CLASS_IDS,
    STEALING_TARGET_CLASSES,
    WEAPON_WEIGHTS_PATH,
    WEAPON_CLASSES,
    WEAPON_CONFIDENCE_THRESHOLD,
    WEAPON_PROXIMITY_THRESHOLD,
    TRACKING_DEFAULT_FPS,
    TRACK_CONFIRMATION_MIN_OBSERVATIONS,
    APPEARANCE_INTERVAL_MAX_GAP_SECONDS,
)
from app.core.weapon_detector import WeaponDetector
from app.core.stealing_detector import StealingObjectDetector
from app.core.object_catalog import ObjectCatalog

logger = logging.getLogger("argus.tracking_pipeline")


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_appearance_intervals(
    trajectory: List[Dict[str, Any]],
    max_gap_seconds: float = APPEARANCE_INTERVAL_MAX_GAP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Computes distinct, non-continuous appearance intervals from timestamped observations.
    Splits into separate intervals when temporal gap between consecutive observations exceeds max_gap_seconds.
    Accurately preserves single-frame observations without inventing durations.
    """
    if not trajectory:
        return []

    # Sort observations chronologically by timestamp
    sorted_pts = sorted(trajectory, key=lambda p: p.get("t", 0.0))
    intervals: List[Dict[str, Any]] = []

    cur_start = sorted_pts[0].get("t", 0.0)
    cur_end = cur_start
    cur_count = 1

    for i in range(1, len(sorted_pts)):
        t = sorted_pts[i].get("t", 0.0)
        gap = t - cur_end
        if gap <= max_gap_seconds:
            cur_end = t
            cur_count += 1
        else:
            dur = round(cur_end - cur_start, 2)
            is_single = bool(cur_start == cur_end or cur_count == 1)
            start_fmt = f"{int(cur_start // 60):02d}:{cur_start % 60:05.2f}"
            end_fmt = f"{int(cur_end // 60):02d}:{cur_end % 60:05.2f}"
            disp = (
                f"{start_fmt} — {end_fmt} ({cur_count} frames)"
                if not is_single
                else f"{start_fmt} (1 frame observation)"
            )
            intervals.append({
                "interval_id": len(intervals) + 1,
                "start_s": round(cur_start, 2),
                "end_s": round(cur_end, 2),
                "duration_s": dur,
                "start_formatted": start_fmt,
                "end_formatted": end_fmt,
                "observation_count": cur_count,
                "is_single_observation": is_single,
        "is_single_frame": is_single,
        "label": disp,
                "is_single_frame": is_single,
                "label": disp,
                "display_text": disp,
            })
            cur_start = t
            cur_end = t
            cur_count = 1

    dur = round(cur_end - cur_start, 2)
    is_single = bool(cur_start == cur_end or cur_count == 1)
    start_fmt = f"{int(cur_start // 60):02d}:{cur_start % 60:05.2f}"
    end_fmt = f"{int(cur_end // 60):02d}:{cur_end % 60:05.2f}"
    disp = (
        f"{start_fmt} — {end_fmt} ({cur_count} frames)"
        if not is_single
        else f"{start_fmt} (1 frame observation)"
    )
    intervals.append({
        "interval_id": len(intervals) + 1,
        "start_s": round(cur_start, 2),
        "end_s": round(cur_end, 2),
        "duration_s": dur,
        "start_formatted": start_fmt,
        "end_formatted": end_fmt,
        "observation_count": cur_count,
        "is_single_observation": is_single,
        "is_single_frame": is_single,
        "label": disp,
        "display_text": disp,
    })

    return intervals


class GeneralizedTracker:
    """
    Unified multi-object tracking, duplicate suppression, and threat association pipeline.
    Reuses Ultralytics ByteTrack infrastructure without duplicate model loading.
    """
    _instance: Optional["GeneralizedTracker"] = None
    _lock = threading.Lock()

    def __init__(self, enable_weapon_detection: bool = True):
        self.device: Any = 0 if torch.cuda.is_available() else "cpu"
        self.general_detector = StealingObjectDetector.get_instance()
        self.enable_weapon_detection = enable_weapon_detection
        self.weapon_detector = WeaponDetector.get_instance() if enable_weapon_detection else None
        self.tracking_fps: float = TRACKING_DEFAULT_FPS
        self.weapon_proximity_threshold: float = WEAPON_PROXIMITY_THRESHOLD

    @classmethod
    def get_instance(cls) -> "GeneralizedTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def reset_tracker(self):
        """
        Explicitly resets tracker state between independent videos or clips.
        Guarantees that track IDs never leak into subsequent videos.
        """
        self.general_detector.reset_tracker()
        if hasattr(self.weapon_detector, "model") and self.weapon_detector.model is not None:
            pred = getattr(self.weapon_detector.model, "predictor", None)
            if pred is not None and hasattr(pred, "trackers") and pred.trackers:
                for t in pred.trackers:
                    if hasattr(t, "reset"):
                        try:
                            t.reset()
                        except Exception:
                            pass
                pred.trackers = []

    def detect_and_track_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        run_weapons: bool = True,
        conf_threshold_general: float = 0.25,
        conf_threshold_weapon: float = WEAPON_CONFIDENCE_THRESHOLD,
        target_class_ids: Optional[List[int]] = None,
        tracker_type: str = "bytetrack.yaml",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Executes unified multi-object tracking across sequential frames:
        1. Tracks all 80 COCO classes (or specified target classes) with ByteTrack.
        2. Detects weapons (pistol, rifle, knife) using dedicated weapon detector.
        3. Tracks weapon instances across frames with persistent ID assignment.
        4. Applies spatial IoU fusion and duplicate suppression between general & weapon detectors.
        5. Preserves frame coordinates, model source traceability, and source timestamps.
        """
        if not frames:
            return [], {}

        # Reset tracker state before starting a new video sequence
        self.reset_tracker()

        h, w = frames[0].shape[:2]
        if h <= 0 or w <= 0:
            return [], {}

        # 1. Run general object detector with ByteTrack tracking (all COCO classes or target subset)
        general_dets, general_summaries = self.general_detector.detect_and_track_frames(
            frames=frames,
            timestamps=timestamps,
            tracker_type=tracker_type,
            conf_threshold=conf_threshold_general,
            filter_class_ids=target_class_ids,
        )

        unified_tracks: Dict[str, Dict[str, Any]] = {}
        for tid, summary in general_summaries.items():
            key = f"OBJ-{tid}"
            summary_copy = dict(summary)
            cname = summary.get("class_name", "object")
            cat = ObjectCatalog._find_category(cname)
            summary_copy["category"] = cat
            summary_copy["entity_type"] = "person" if summary.get("is_person") else "object"
            summary_copy["is_weapon"] = False
            summary_copy["track_id_str"] = key
            summary_copy["model_source"] = "coco_yolov8n"
            summary_copy["is_suppressed"] = False

            # Appearance intervals
            traj = summary_copy.get("bbox_trajectory", [])
            intervals = compute_appearance_intervals(traj)
            summary_copy["appearance_intervals"] = intervals

            # Class-specific confidence and temporal consistency validation
            req_thresh = ObjectCatalog.get_class_threshold(cname)
            summary_copy["calibrated_threshold"] = req_thresh
            obs_count = len(traj)
            max_conf = summary_copy.get("max_confidence", 0.0)

            if obs_count >= TRACK_CONFIRMATION_MIN_OBSERVATIONS and max_conf >= req_thresh:
                summary_copy["detection_status"] = "confirmed_track"
            else:
                summary_copy["detection_status"] = "uncertain_candidate"
                summary_copy["candidate_reason"] = (
                    "Single-frame observation" if obs_count == 1
                    else f"Confidence ({max_conf:.2f}) below calibrated threshold ({req_thresh:.2f})"
                )

            unified_tracks[key] = summary_copy

        all_detections = list(general_dets)
        for d in all_detections:
            d["is_weapon"] = False
            d["entity_type"] = "person" if d.get("is_person") else "object"
            d["track_id_str"] = f"OBJ-{d['track_id']}" if d.get("track_id") is not None else None
            d["model_source"] = "coco_yolov8n"
            d["is_suppressed"] = False
            d["category"] = ObjectCatalog._find_category(d.get("class_name", "object"))

        # 2. Run dedicated weapon detector if enabled
        weapon_detections_list: List[Dict[str, Any]] = []
        if run_weapons:
            self.weapon_detector.load_model()
            weapon_track_counter = 1
            active_weapon_tracks: Dict[int, Dict[str, Any]] = {}

            for f_idx, (frame, t_sec) in enumerate(zip(frames, timestamps)):
                w_dets = self.weapon_detector.detect_frame(frame, conf_threshold=conf_threshold_weapon)
                for wd in w_dets:
                    wd["frame_index"] = f_idx
                    wd["timestamp"] = round(t_sec, 3)
                    wd["entity_type"] = "weapon"
                    wd["model_source"] = "weapon_shield_yolov8s"
                    wd["category"] = "Weapons"
                    wd["is_suppressed"] = False

                    # Spatial-temporal association for weapon tracks (IoU / distance continuity)
                    matched_tid = None
                    cx, cy = wd["normalized_centroid"]

                    for w_id, w_rec in active_weapon_tracks.items():
                        if w_rec["class_name"] == wd["class_name"] and (f_idx - w_rec["last_frame"]) <= 2:
                            prev_cx, prev_cy = w_rec["centroids"][-1]
                            dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                            if dist < 0.15:
                                matched_tid = w_id
                                break

                    if matched_tid is None:
                        matched_tid = weapon_track_counter
                        weapon_track_counter += 1
                        key = f"WEP-{matched_tid}"
                        active_weapon_tracks[matched_tid] = {
                            "track_id": matched_tid,
                            "track_id_str": key,
                            "class_name": wd["class_name"],
                            "category": "Weapons",
                            "model_source": "weapon_shield_yolov8s",
                            "is_person": False,
                            "is_target_object": False,
                            "is_weapon": True,
                            "is_suppressed": False,
                            "entity_type": "weapon",
                            "first_seen_s": round(t_sec, 2),
                            "last_seen_s": round(t_sec, 2),
                            "first_frame": f_idx,
                            "last_frame": f_idx,
                            "detection_count": 0,
                            "max_confidence": wd["confidence"],
                            "centroids": [],
                            "raw_bboxes_norm": [],
                            "raw_confidences": [],
                        }

                    w_rec = active_weapon_tracks[matched_tid]
                    w_rec["last_seen_s"] = round(t_sec, 2)
                    w_rec["last_frame"] = f_idx
                    w_rec["detection_count"] += 1
                    w_rec["max_confidence"] = max(w_rec["max_confidence"], wd["confidence"])
                    w_rec["centroids"].append([cx, cy])
                    w_rec["raw_bboxes_norm"].append({"t": round(t_sec, 3), "bbox_norm": wd["normalized_bbox"]})
                    w_rec["raw_confidences"].append({"t": round(t_sec, 3), "conf": wd["confidence"]})

                    wd["track_id"] = matched_tid
                    wd["track_id_str"] = f"WEP-{matched_tid}"
                    weapon_detections_list.append(wd)

            # Consolidate weapon track summaries with appearance intervals and validation
            for w_id, w_info in active_weapon_tracks.items():
                duration = round(w_info["last_seen_s"] - w_info["first_seen_s"], 2)
                centroids = w_info["centroids"]
                displacement = 0.0
                if len(centroids) >= 2:
                    dx = centroids[-1][0] - centroids[0][0]
                    dy = centroids[-1][1] - centroids[0][1]
                    displacement = math.sqrt(dx * dx + dy * dy)

                expected_frames = w_info["last_frame"] - w_info["first_frame"] + 1
                gap_count = max(0, expected_frames - w_info["detection_count"])

                conf_vals = [c["conf"] for c in w_info["raw_confidences"]]
                avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else w_info["max_confidence"]

                # Appearance intervals
                traj = list(w_info["raw_bboxes_norm"])
                intervals = compute_appearance_intervals(traj)

                # Class-specific confidence and temporal consistency validation
                cname = w_info["class_name"]
                req_thresh = ObjectCatalog.get_class_threshold(cname)
                obs_count = w_info["detection_count"]
                max_conf = w_info["max_confidence"]

                if obs_count >= TRACK_CONFIRMATION_MIN_OBSERVATIONS and max_conf >= req_thresh:
                    det_status = "confirmed_track"
                    cand_reason = None
                elif obs_count == 1 and max_conf >= 0.70:
                    det_status = "confirmed_track"
                    cand_reason = "Single observation with high confidence (>=0.70)"
                else:
                    det_status = "uncertain_candidate"
                    cand_reason = (
                        "Isolated observation or sub-threshold confidence (potential shadow, hand fold, or texture)"
                    )

                key = w_info["track_id_str"]
                unified_tracks[key] = {
                    "track_id": w_info["track_id"],
                    "track_id_str": key,
                    "class_name": cname,
                    "category": "Weapons",
                    "model_source": "weapon_shield_yolov8s",
                    "is_person": False,
                    "is_target_object": False,
                    "is_weapon": True,
                    "is_suppressed": False,
                    "entity_type": "weapon",
                    "first_seen_s": w_info["first_seen_s"],
                    "last_seen_s": w_info["last_seen_s"],
                    "duration_s": duration,
                    "detection_count": w_info["detection_count"],
                    "gap_count": gap_count,
                    "max_confidence": round(w_info["max_confidence"], 4),
                    "avg_confidence": round(avg_conf, 4),
                    "calibrated_threshold": req_thresh,
                    "detection_status": det_status,
                    "candidate_reason": cand_reason,
                    "total_displacement": round(displacement, 4),
                    "is_moving": bool(displacement > 0.05),
                    "bbox_trajectory": traj,
                    "confidence_history": list(w_info["raw_confidences"]),
                    "appearance_intervals": intervals,
                }

        # 3. Model Fusion & Duplicate Detection Suppression (Section 9)
        # Suppress overlapping generic COCO detections (e.g. COCO knife) when dedicated weapon detector is active
        suppressed_general_keys = set()
        for wd in weapon_detections_list:
            f_idx = wd["frame_index"]
            w_box = wd["bbox"]
            for gd in all_detections:
                if gd["frame_index"] == f_idx:
                    g_box = gd["bbox"]
                    iou = compute_box_iou(g_box, w_box)
                    # If high overlap with a weapon detection
                    if iou >= 0.40:
                        g_name = gd.get("class_name", "").lower()
                        # Direct knife overlap or generic object misclassification on firearm
                        if g_name in ["knife", "cell phone", "remote", "scissors"]:
                            gd["is_suppressed"] = True
                            gd["suppressed_by"] = wd["track_id_str"]
                            wd["fused_with"] = f"COCO-{gd['class_name']}"
                            if gd.get("track_id_str"):
                                suppressed_general_keys.add(gd["track_id_str"])

        # Update suppressed status on tracks where all detections were suppressed
        for key in suppressed_general_keys:
            if key in unified_tracks:
                track_dets = [d for d in all_detections if d.get("track_id_str") == key]
                if track_dets and all(d.get("is_suppressed") for d in track_dets):
                    unified_tracks[key]["is_suppressed"] = True
                    unified_tracks[key]["suppression_reason"] = "Subsumed by dedicated tactical weapon detection"

        # Combine weapon detections into all_detections
        all_detections.extend(weapon_detections_list)

        # Clear predictor state after processing
        self.reset_tracker()
        return all_detections, unified_tracks

    def evaluate_person_weapon_association(
        self,
        detections: List[Dict[str, Any]],
        track_summaries: Dict[str, Dict[str, Any]],
        proximity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates potential spatial and temporal associations between detected persons and weapons.
        Provides uncalibrated assistive cues for human security review.
        """
        prox_thresh = proximity_threshold if proximity_threshold is not None else self.weapon_proximity_threshold

        if isinstance(track_summaries, dict):
            all_tracks = list(track_summaries.values())
        elif isinstance(track_summaries, list):
            all_tracks = track_summaries
        else:
            all_tracks = []

        person_tracks = [t for t in all_tracks if t.get("is_person") and not t.get("is_suppressed")]
        weapon_tracks = [t for t in all_tracks if t.get("is_weapon") and not t.get("is_suppressed")]

        if not weapon_tracks:
            return {
                "status": "NO_WEAPON_DETECTED",
                "cues": [],
                "weapon_count": 0,
                "weapons_detected": [],
                "review_recommendation": "STANDARD_REVIEW",
                "decision_reason": "No firearms, bladed weapons, or tactical threats detected in analyzed footage.",
                "responsible_ai_disclaimer": "AI detection triage indicator. Negative detection does not constitute legal certification of absence.",
            }

        unique_weapon_classes = sorted(list({t["class_name"] for t in weapon_tracks}))

        # Group detections by frame index
        frames_map: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            if not d.get("is_suppressed"):
                frames_map.setdefault(d["frame_index"], []).append(d)

        association_cues: List[Dict[str, Any]] = []

        for w_track in weapon_tracks:
            w_key = w_track.get("track_id_str", f"WEP-{w_track['track_id']}")
            w_class = w_track["class_name"]

            for p_track in person_tracks:
                p_key = p_track.get("track_id_str", f"OBJ-{p_track['track_id']}")
                p_id = p_track["track_id"]
                distances = []
                common_timestamps = []

                for f_idx, frame_dets in frames_map.items():
                    p_det = next((d for d in frame_dets if d.get("is_person") and d.get("track_id") == p_id), None)
                    w_det = next((d for d in frame_dets if d.get("is_weapon") and d.get("track_id_str") == w_key), None)
                    if p_det and w_det:
                        p_cx, p_cy = p_det["normalized_centroid"]
                        w_cx, w_cy = w_det["normalized_centroid"]
                        dist = math.sqrt((p_cx - w_cx) ** 2 + (p_cy - w_cy) ** 2)
                        distances.append(dist)
                        common_timestamps.append(p_det["timestamp"])

                if distances:
                    min_dist = min(distances)
                    if min_dist <= prox_thresh:
                        start_t = min(common_timestamps)
                        end_t = max(common_timestamps)
                        contact_duration = round(end_t - start_t, 2)
                        contact_frames = sum(1 for d in distances if d <= prox_thresh)

                        association_cues.append({
                            "cue_type": "person_weapon_proximity",
                            "person_track_id": p_key,
                            "weapon_track_id": w_key,
                            "weapon_class": w_class,
                            "min_distance": round(min_dist, 3),
                            "contact_duration_s": contact_duration,
                            "contact_frames": contact_frames,
                            "start_s": start_t,
                            "end_s": end_t,
                            "description": (
                                f"Spatial Proximity Cue: {w_class} (Track {w_key}) observed in close proximity "
                                f"to Person (Track {p_key}) for {contact_duration}s (min norm distance: {min_dist:.2f})."
                            ),
                        })

        # Separate confirmed persistent weapon tracks from single-frame uncertain glitches
        confirmed_weapon_tracks = [
            t for t in weapon_tracks
            if t.get("detection_status") == "confirmed_track" or t.get("max_confidence", 0.0) >= 0.70
        ]
        uncertain_weapon_tracks = [
            t for t in weapon_tracks if t not in confirmed_weapon_tracks
        ]

        if not confirmed_weapon_tracks and uncertain_weapon_tracks:
            unique_uncertain = sorted(list({t["class_name"] for t in uncertain_weapon_tracks}))
            max_conf = max([t["max_confidence"] for t in uncertain_weapon_tracks], default=0.0)
            return {
                "status": "WEAPON_DETECTION_UNCERTAIN",
                "cues": association_cues,
                "weapon_count": len(uncertain_weapon_tracks),
                "weapons_detected": unique_uncertain,
                "confirmed_weapons": [],
                "uncertain_weapons": unique_uncertain,
                "max_confidence": round(max_conf, 4),
                "review_recommendation": "STANDARD_REVIEW",
                "decision_reason": (
                    f"Tentative candidate observation(s) ({', '.join(unique_uncertain)}) below persistence/confidence threshold. "
                    "Likely visual artifact, hand fold, or background shadow; review raw overlay if required."
                ),
                "responsible_ai_disclaimer": (
                    "AI-assisted triage observation. Spatial proximity does not constitute proof of weapon possession."
                ),
            }

        max_weapon_conf = max([t["max_confidence"] for t in confirmed_weapon_tracks], default=0.0)
        has_sustained = any(t["detection_count"] >= 3 for t in confirmed_weapon_tracks)

        if has_sustained or max_weapon_conf >= 0.60:
            status = "WEAPON_TRACKED_SUSTAINED"
            recommendation = "PRIORITY_HUMAN_REVIEW"
        elif max_weapon_conf >= 0.45:
            status = "POTENTIAL_WEAPON_DETECTED"
            recommendation = "ELEVATED_REVIEW"
        else:
            status = "WEAPON_DETECTION_UNCERTAIN"
            recommendation = "STANDARD_REVIEW"

        if association_cues:
            reason = (
                f"Potential {', '.join(unique_weapon_classes)} detected with concurrent person association "
                f"({len(association_cues)} proximity observation{'s' if len(association_cues) > 1 else ''})."
            )
        else:
            reason = f"Potential {', '.join(unique_weapon_classes)} detected in scene. Review video overlay."

        return {
            "status": status,
            "cues": association_cues,
            "weapon_count": len(weapon_tracks),
            "weapons_detected": unique_weapon_classes,
            "confirmed_weapons": sorted(list({t["class_name"] for t in confirmed_weapon_tracks})),
            "uncertain_weapons": sorted(list({t["class_name"] for t in uncertain_weapon_tracks})),
            "max_confidence": round(max_weapon_conf if confirmed_weapon_tracks else 0.0, 4),
            "review_recommendation": recommendation,
            "decision_reason": reason,
            "responsible_ai_disclaimer": (
                "AI-assisted triage observation. Spatial proximity does not constitute proof "
                "of ownership, brandishing, or weapon possession. Operator verification required."
            ),
        }

    def analyze_clip(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = 10.0,
        context_buffer_seconds: float = 1.0,
        sample_fps: float = TRACKING_DEFAULT_FPS,
        run_weapons: bool = True,
        target_classes: Optional[List[str]] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        fps: Optional[float] = None,
        target_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Decoupled clip analysis interface.
        Extracts candidate interval with context buffer, samples frames efficiently,
        runs multi-object tracking across all 80 COCO classes + weapons,
        and returns complete grounded telemetry including object inventory.
        """
        if start_s is not None:
            start_time = start_s
        if end_s is not None:
            end_time = end_s
        if fps is not None:
            sample_fps = fps
        if target_fps is not None:
            sample_fps = target_fps

        t0 = time.perf_counter()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "error": f"Unable to open video: {video_path}",
                "clip_start_s": round(start_time, 2),
                "review_recommendation": "INSUFFICIENT_VISUAL_EVIDENCE",
                "limitations": "Video could not be opened for inspection.",
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        buffered_start = max(0.0, start_time - context_buffer_seconds)
        buffered_end = min(duration, end_time + context_buffer_seconds)
        if buffered_end <= buffered_start:
            buffered_end = min(duration, buffered_start + 1.0)

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
                frames.append(frame)
                timestamps.append(st)

        cap.release()

        if not frames:
            return {
                "status": "no_frames_extracted",
                "clip_start_s": round(start_time, 2),
                "review_recommendation": "INSUFFICIENT_VISUAL_EVIDENCE",
                "limitations": "Zero frames retrieved in target interval.",
            }

        # Map target_classes to COCO IDs if provided
        target_class_ids = None
        filter_weapon = run_weapons
        if target_classes:
            target_lower = {c.strip().lower() for c in target_classes}
            # Check if weapons are in target classes
            has_wep = any(w.lower() in target_lower for w in ["pistol", "rifle", "knife", "weapon"])
            filter_weapon = run_weapons and has_wep
            # Map COCO classes
            self.general_detector.load_model()
            target_class_ids = [
                cid for cid, name in self.general_detector.supported_classes.items()
                if name.lower() in target_lower
            ]

        try:
            detections, track_summaries = self.detect_and_track_frames(
                frames=frames,
                timestamps=timestamps,
                run_weapons=filter_weapon,
                target_class_ids=target_class_ids,
            )
            weapon_eval = self.evaluate_person_weapon_association(detections, track_summaries)
            proc_time = round(time.perf_counter() - t0, 3)

            # Active (non-suppressed) tracks
            visible_tracks = [t for t in track_summaries.values() if not t.get("is_suppressed")]
            objects_detected = sorted(list({t["class_name"] for t in visible_tracks}))

            # Build object inventory
            inventory_data = ObjectCatalog.build_inventory(visible_tracks)

            return {
                "status": "completed",
                "enabled": True,
                "tracker_type": "ByteTrack",
                "clip_start_s": round(start_time, 2),
                "analyzed_interval": {
                    "start_time": round(buffered_start, 2),
                    "end_time": round(buffered_end, 2),
                    "context_buffer_seconds": context_buffer_seconds,
                    "frames_analyzed": len(frames),
                    "sample_fps": sample_fps,
                    "video_duration_s": round(duration, 2),
                },
                "objects_detected": objects_detected,
                "track_summaries": list(track_summaries.values()),
                "visible_tracks": visible_tracks,
                "inventory": inventory_data,
                "detections_count": len(detections),
                "weapon_analysis": weapon_eval,
                "processing_time_s": proc_time,
            }

        except Exception as e:
            logger.error(f"Error during generalized tracking analysis: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "clip_start_s": round(start_time, 2),
                "review_recommendation": "MANUAL_REVIEW_REQUIRED",
                "limitations": f"Tracker encountered error: {e}",
                "processing_time_s": round(time.perf_counter() - t0, 3),
            }

    # Alias for convenience
    track_video = analyze_clip

    def targeted_scan(
        self,
        video_path: str,
        target_classes: List[str],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        sample_fps: float = 10.0,
        conf_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Executes a targeted scan across video footage for specific object classes (Mode B).
        Accurately reports frames examined, sampling rate, and coverage percentage.
        """
        t0 = time.perf_counter()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": f"Cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        scan_start = max(0.0, float(start_time))
        scan_end = min(duration, float(end_time)) if end_time is not None else duration
        if scan_end <= scan_start:
            scan_end = min(duration, scan_start + 1.0)

        step_s = 1.0 / max(1.0, float(sample_fps))
        sample_times = np.arange(scan_start, scan_end, step_s).tolist()
        if not sample_times:
            sample_times = [scan_start]

        frames: List[np.ndarray] = []
        timestamps: List[float] = []

        for st in sample_times:
            frame_num = min(total_frames - 1, int(st * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                timestamps.append(st)

        cap.release()

        frames_in_interval = int((scan_end - scan_start) * fps)
        coverage_pct = round(100.0 * len(frames) / max(1, frames_in_interval), 1)

        # Check target class support
        target_lower = [c.strip().lower() for c in target_classes]
        all_catalog = ObjectCatalog.get_supported_classes()
        supported_names = {c["normalized_name"] for c in all_catalog}
        supported_targets = [c for c in target_lower if c in supported_names]
        unsupported_targets = [c for c in target_lower if c not in supported_names]

        # Determine if weapon models should run
        weapon_names = {"pistol", "rifle", "knife"}
        run_weapons = any(c in weapon_names for c in supported_targets)

        # Map to COCO IDs
        self.general_detector.load_model()
        target_coco_ids = [
            cid for cid, name in self.general_detector.supported_classes.items()
            if name.lower() in supported_targets
        ]

        # Execute tracking
        detections, track_summaries = self.detect_and_track_frames(
            frames=frames,
            timestamps=timestamps,
            run_weapons=run_weapons,
            conf_threshold_general=conf_threshold,
            target_class_ids=target_coco_ids if target_coco_ids else None,
        )

        # Filter tracks matching target_classes
        matching_tracks = [
            t for t in track_summaries.values()
            if t["class_name"].lower() in supported_targets and not t.get("is_suppressed")
        ]
        matching_detections = [
            d for d in detections
            if d.get("class_name", "").lower() in supported_targets and not d.get("is_suppressed")
        ]

        proc_time = round(time.perf_counter() - t0, 3)

        return {
            "status": "completed",
            "video_source": Path(video_path).name,
            "target_classes_requested": target_classes,
            "supported_classes": supported_targets,
            "unsupported_classes": unsupported_targets,
            "scan_metadata": {
                "start_time_s": round(scan_start, 2),
                "end_time_s": round(scan_end, 2),
                "video_duration_s": round(duration, 2),
                "frames_examined": len(frames),
                "total_interval_frames": frames_in_interval,
                "sample_fps": sample_fps,
                "sampling_coverage_pct": coverage_pct,
            },
            "matches_found": len(matching_tracks) > 0,
            "matched_tracks_count": len(matching_tracks),
            "matched_detections_count": len(matching_detections),
            "matching_tracks": matching_tracks,
            "inventory": ObjectCatalog.build_inventory(matching_tracks),
            "processing_time_s": proc_time,
        }

    def search_objects(
        self,
        track_summaries: List[Dict[str, Any]],
        query: str,
        status_filter: str = "all",
    ) -> Dict[str, Any]:
        """
        Fast search across existing analyzed tracks and detections (Mode A).
        Uses natural language alias resolution, computes appearance intervals,
        and explicitly distinguishes matches vs supported-not-detected vs unsupported classes.
        """
        q = (query or "").strip().lower()
        visible_tracks = [t for t in track_summaries if not t.get("is_suppressed")]

        # Ensure appearance intervals are populated on all tracks
        for t in visible_tracks:
            if "appearance_intervals" not in t:
                t["appearance_intervals"] = compute_appearance_intervals(t.get("bbox_trajectory", []))

        if not q:
            filtered = visible_tracks
            if status_filter == "confirmed_only":
                filtered = [t for t in filtered if t.get("detection_status") == "confirmed_track"]
            elif status_filter == "uncertain_only":
                filtered = [t for t in filtered if t.get("detection_status") == "uncertain_candidate"]

            return {
                "query": query,
                "status": "all_objects",
                "is_supported_class": True,
                "canonical_classes": [],
                "matched_aliases": [],
                "matched_count": len(filtered),
                "matching_tracks": filtered,
                "confirmed_tracks_count": sum(1 for m in filtered if m.get("detection_status") == "confirmed_track"),
                "uncertain_candidates_count": sum(1 for m in filtered if m.get("detection_status") == "uncertain_candidate"),
                "explanation": "Displaying all tracked objects.",
                "message": "Returned all tracked objects.",
            }

        # Resolve query with natural language alias catalog
        resolved = ObjectCatalog.resolve_query(q)
        is_supported = resolved["is_supported"]
        canonical_targets = {c.lower() for c in resolved["canonical_classes"]}

        matched_tracks = []
        for t in visible_tracks:
            cname = t.get("class_name", "").lower()
            tid_str = t.get("track_id_str", "").lower()
            cat = t.get("category", "").lower()

            matches_target = (
                cname in canonical_targets
                or q == cname
                or q in cname
                or cname in q
                or q in tid_str
                or q in cat
            )
            if matches_target:
                matched_tracks.append(t)

        # Apply status filter
        if status_filter == "confirmed_only":
            filtered_matches = [t for t in matched_tracks if t.get("detection_status") == "confirmed_track"]
        elif status_filter == "uncertain_only":
            filtered_matches = [t for t in matched_tracks if t.get("detection_status") == "uncertain_candidate"]
        else:
            filtered_matches = matched_tracks

        confirmed_cnt = sum(1 for m in filtered_matches if m.get("detection_status") == "confirmed_track")
        uncertain_cnt = sum(1 for m in filtered_matches if m.get("detection_status") == "uncertain_candidate")

        if filtered_matches:
            status = "matches_found"
            msg = f"Found {len(filtered_matches)} track(s) matching '{query}' ({resolved['explanation']})."
        elif is_supported:
            status = "supported_not_detected"
            msg = f"{resolved['explanation']} Zero instances detected in this video interval."
        else:
            status = "unsupported_class"
            msg = f"Class '{query}' is not recognized by active models (YOLOv8 COCO 80 + WeaponShield 3)."

        return {
            "query": query,
            "status": status,
            "is_supported_class": is_supported,
            "canonical_classes": resolved["canonical_classes"],
            "matched_aliases": resolved["matched_aliases"],
            "matched_count": len(filtered_matches),
            "matching_tracks": filtered_matches,
            "confirmed_tracks_count": confirmed_cnt,
            "uncertain_candidates_count": uncertain_cnt,
            "explanation": resolved["explanation"],
            "message": msg,
        }


def evaluate_person_weapon_association(
    person_tracks_or_detections: Any,
    weapon_tracks_or_summaries: Any,
    proximity_threshold: float = WEAPON_PROXIMITY_THRESHOLD,
) -> Dict[str, Any]:
    """Module-level convenience wrapper for GeneralizedTracker.evaluate_person_weapon_association."""
    tracker = GeneralizedTracker.get_instance()
    if isinstance(person_tracks_or_detections, list) and isinstance(weapon_tracks_or_summaries, list):
        combined = person_tracks_or_detections + weapon_tracks_or_summaries
        return tracker.evaluate_person_weapon_association([], combined, proximity_threshold)
    return tracker.evaluate_person_weapon_association(person_tracks_or_detections, weapon_tracks_or_summaries, proximity_threshold)
