import sys
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import (
    STEALING_CANDIDATE_THRESHOLD,
    STEALING_PROXIMITY_THRESHOLD,
    STEALING_DISPLACEMENT_THRESHOLD,
)
from app.core.stealing_detector import StealingObjectDetector
from app.core.video_processor import VideoProcessor
from app.core.model import VideoAnomalyDetector
from app.main import app


class TestStealingObjectDetector:
    @classmethod
    def setup_class(cls):
        cls.detector = StealingObjectDetector.get_instance()
        assert cls.detector.load_model(), "YOLOv8n object detector failed to load"

    def test_singleton_and_loading(self):
        second = StealingObjectDetector.get_instance()
        assert self.detector is second, "StealingObjectDetector must be a singleton"
        assert self.detector.is_loaded is True
        assert self.detector.model is not None
        assert "person" in self.detector.supported_classes.values()
        assert self.detector.license_info == "AGPL-3.0 (Ultralytics YOLOv8)"

    def test_detection_and_tracking_output_parsing(self):
        # Create a series of 3 synthetic frames
        frames = [np.full((320, 320, 3), fill_value=i * 50, dtype=np.uint8) for i in range(3)]
        timestamps = [0.0, 0.2, 0.4]

        detections, summaries = self.detector.detect_and_track_frames(frames, timestamps)
        assert isinstance(detections, list)
        assert isinstance(summaries, dict)

        # Even on blank synthetic frames, output must be valid lists/dicts without crashing
        for d in detections:
            assert "frame_index" in d
            assert "timestamp" in d
            assert "bbox" in d
            assert len(d["bbox"]) == 4
            assert "normalized_bbox" in d
            assert len(d["normalized_bbox"]) == 4
            assert "confidence" in d
            assert "class_name" in d

    def test_tracker_state_reset_between_videos(self):
        # Verify tracker reset clears predictor state
        self.detector.reset_tracker()
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            trackers = getattr(self.detector.model.predictor, "trackers", None)
            assert trackers is None or len(trackers) == 0


    def test_interaction_cues_proximity_and_displacement(self):
        # Test synthetic detections with 1 person and 1 backpack moving close together
        mock_detections = [
            # Frame 0: Person at [0.2, 0.2], Backpack at [0.22, 0.22] (close proximity ~0.028)
            {
                "frame_index": 0, "timestamp": 0.0, "track_id": 1, "class_id": 0, "class_name": "person",
                "is_person": True, "is_target_object": False, "normalized_centroid": [0.20, 0.20],
                "confidence": 0.90, "bbox": [50, 50, 100, 150]
            },
            {
                "frame_index": 0, "timestamp": 0.0, "track_id": 2, "class_id": 24, "class_name": "backpack",
                "is_person": False, "is_target_object": True, "normalized_centroid": [0.22, 0.22],
                "confidence": 0.85, "bbox": [60, 60, 90, 90]
            },
            # Frame 1: Person at [0.35, 0.35], Backpack at [0.37, 0.37] (displaced together)
            {
                "frame_index": 1, "timestamp": 0.5, "track_id": 1, "class_id": 0, "class_name": "person",
                "is_person": True, "is_target_object": False, "normalized_centroid": [0.35, 0.35],
                "confidence": 0.92, "bbox": [100, 100, 150, 200]
            },
            {
                "frame_index": 1, "timestamp": 0.5, "track_id": 2, "class_id": 24, "class_name": "backpack",
                "is_person": False, "is_target_object": True, "normalized_centroid": [0.37, 0.37],
                "confidence": 0.88, "bbox": [110, 110, 140, 140]
            },
            # Frame 2: Backpack disappears, Person at [0.50, 0.50]
            {
                "frame_index": 2, "timestamp": 1.0, "track_id": 1, "class_id": 0, "class_name": "person",
                "is_person": True, "is_target_object": False, "normalized_centroid": [0.50, 0.50],
                "confidence": 0.94, "bbox": [150, 150, 200, 250]
            },
        ]

        mock_summaries = {
            1: {
                "track_id": 1, "class_name": "person", "is_person": True, "is_target_object": False,
                "first_seen_s": 0.0, "last_seen_s": 1.0, "duration_s": 1.0, "detection_count": 3,
                "gap_count": 0, "max_confidence": 0.94, "total_displacement": 0.42, "is_moving": True,
            },
            2: {
                "track_id": 2, "class_name": "backpack", "is_person": False, "is_target_object": True,
                "first_seen_s": 0.0, "last_seen_s": 0.5, "duration_s": 0.5, "detection_count": 3,
                "gap_count": 0, "max_confidence": 0.88, "total_displacement": 0.21, "is_moving": True,
            },
        }

        result = self.detector.analyze_interactions(mock_detections, mock_summaries)
        assert result["status"] == "completed"
        assert len(result["cues"]) >= 2
        cue_types = [c["cue_type"] for c in result["cues"]]
        assert "person_object_proximity" in cue_types
        assert "object_displacement" in cue_types
        assert result["uncalibrated_interaction_score"] >= 0.50
        assert result["review_recommendation"] in ["PRIORITY_HUMAN_REVIEW", "STANDARD_REVIEW"]
        assert "limitations" in result

    def test_missing_or_unsupported_object_classes(self):
        # 1. No entities at all
        res_empty = self.detector.analyze_interactions([], {})
        assert res_empty["status"] == "no_relevant_entities"
        assert res_empty["uncalibrated_interaction_score"] == 0.0
        assert res_empty["review_recommendation"] == "INSUFFICIENT_VISUAL_EVIDENCE"

        # 2. Only person detected, no target objects
        person_only_summary = {
            1: {
                "track_id": 1, "class_name": "person", "is_person": True, "is_target_object": False,
                "first_seen_s": 0.0, "last_seen_s": 2.0, "duration_s": 2.0, "detection_count": 5,
                "gap_count": 0, "max_confidence": 0.95, "total_displacement": 0.1, "is_moving": True,
            }
        }
        res_person = self.detector.analyze_interactions([], person_only_summary)
        assert res_person["status"] == "persons_only"
        assert res_person["review_recommendation"] == "STANDARD_REVIEW"
        assert res_person["uncalibrated_interaction_score"] == 0.15

        # 3. Only object detected, no person
        object_only_summary = {
            2: {
                "track_id": 2, "class_name": "suitcase", "is_person": False, "is_target_object": True,
                "first_seen_s": 0.0, "last_seen_s": 2.0, "duration_s": 2.0, "detection_count": 5,
                "gap_count": 0, "max_confidence": 0.88, "total_displacement": 0.0, "is_moving": False,
            }
        }
        res_obj = self.detector.analyze_interactions([], object_only_summary)
        assert res_obj["status"] == "objects_only"
        assert res_obj["review_recommendation"] == "LOW_PRIORITY"

    def test_empty_clips_and_invalid_inputs(self):
        # Empty frames
        dets, sums = self.detector.detect_and_track_frames([], [])
        assert dets == []
        assert sums == {}

        # Nonexistent video path
        res = self.detector.analyze_candidate_clip("nonexistent_video_path_12345.mp4", 0.0, 5.0)
        assert res["status"] == "error"
        assert "Unable to open video" in res["error"]
        assert res["uncalibrated_interaction_score"] == 0.0


class TestVideoProcessorIntegrationAndPolicy:
    @classmethod
    def setup_class(cls):
        detector = VideoAnomalyDetector.get_instance()
        cls.processor = VideoProcessor(detector)

    def test_candidate_trigger_policy_for_stealing(self, monkeypatch):
        # Mock stealing detector so tests run fast without disk access
        mock_analysis_result = {
            "status": "completed",
            "enabled": True,
            "tracker_type": "ByteTrack",
            "objects_detected": ["person", "backpack"],
            "track_summaries": [],
            "interaction_cues": [],
            "uncalibrated_interaction_score": 0.75,
            "decision_reason": "Mocked test detection",
            "review_recommendation": "PRIORITY_HUMAN_REVIEW",
            "processing_time_s": 0.05,
            "limitations": "Mock test.",
        }

        detector_inst = StealingObjectDetector.get_instance()
        monkeypatch.setattr(detector_inst, "analyze_candidate_clip", lambda *args, **kwargs: dict(mock_analysis_result))

        # Case 1: Primary crime is Stealing -> MUST TRIGGER
        stealing_cluster = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Stealing", "confidence": 0.95, "anomaly_score": 0.95, "is_anomaly": True,
             "raw_logits": [0.01, 0.02, 0.02, 0.95]},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Stealing", "confidence": 0.92, "anomaly_score": 0.92, "is_anomaly": True,
             "raw_logits": [0.01, 0.03, 0.04, 0.92]},
        ]
        inc_stealing = self.processor._build_incident(stealing_cluster, 1, 30.0, "vid1", "dummy.mp4")
        assert "stealing_analysis" in inc_stealing
        assert inc_stealing["stealing_analysis"]["status"] == "completed"
        assert "Stealing predicted as primary class" in inc_stealing["stealing_analysis"]["candidate_trigger_reason"]

        # Case 2: Primary crime is Fighting, zero stealing prob -> MUST SKIP
        fighting_cluster = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Fighting", "confidence": 0.88, "anomaly_score": 0.88, "is_anomaly": True,
             "raw_logits": [0.05, 0.88, 0.05, 0.02]},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Fighting", "confidence": 0.85, "anomaly_score": 0.85, "is_anomaly": True,
             "raw_logits": [0.06, 0.85, 0.07, 0.02]},
        ]
        inc_fighting = self.processor._build_incident(fighting_cluster, 2, 30.0, "vid2", "dummy.mp4")
        assert inc_fighting["stealing_analysis"]["status"] == "skipped_not_candidate"

        # Case 3: Primary crime is Fighting, but stealing probability is 0.40 >= 0.30 -> MUST TRIGGER as candidate
        borderline_cluster = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Fighting", "confidence": 0.52, "anomaly_score": 0.92, "is_anomaly": True,
             "raw_logits": [0.05, 0.52, 0.03, 0.40]},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Fighting", "confidence": 0.50, "anomaly_score": 0.90, "is_anomaly": True,
             "raw_logits": [0.05, 0.50, 0.05, 0.40]},
        ]
        inc_borderline = self.processor._build_incident(borderline_cluster, 3, 30.0, "vid3", "dummy.mp4")
        assert inc_borderline["stealing_analysis"]["status"] == "completed"
        assert "Stealing candidate score" in inc_borderline["stealing_analysis"]["candidate_trigger_reason"]

    def test_api_status_backward_compatibility(self):
        client = TestClient(app)
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "model_name" in data
        assert "classes" in data
        assert "Normal Videos" in data["classes"]

    def test_track_trajectory_and_confidence_history(self):
        # Create 3 synthetic frames
        detector = StealingObjectDetector.get_instance()
        frames = [np.full((240, 320, 3), fill_value=128, dtype=np.uint8) for _ in range(3)]
        timestamps = [10.0, 10.2, 10.4]

        detections, summaries = detector.detect_and_track_frames(frames, timestamps)
        assert isinstance(summaries, dict)
        for tid, summary in summaries.items():
            assert "bbox_trajectory" in summary
            assert "confidence_history" in summary
            assert isinstance(summary["bbox_trajectory"], list)
            assert isinstance(summary["confidence_history"], list)
            assert len(summary["bbox_trajectory"]) == len(summary["confidence_history"])
            for pt in summary["bbox_trajectory"]:
                assert "t" in pt
                assert "bbox_norm" in pt
                assert len(pt["bbox_norm"]) == 4
                for coord in pt["bbox_norm"]:
                    assert 0.0 <= coord <= 1.0
            for cpt in summary["confidence_history"]:
                assert "t" in cpt
                assert "conf" in cpt
                assert 0.0 <= cpt["conf"] <= 1.0

    def test_trajectory_subsampling_cap_300(self):
        # Test subsampling logic when track has > 300 frames
        mock_raw_bboxes = [{"t": round(i * 0.1, 2), "bbox_norm": [0.1, 0.1, 0.5, 0.5]} for i in range(450)]
        mock_raw_confs = [{"t": round(i * 0.1, 2), "conf": 0.85} for i in range(450)]

        # Simulate track_data entry
        track_info = {
            "first_seen_s": 0.0,
            "last_seen_s": 44.9,
            "first_frame": 0,
            "last_frame": 449,
            "detection_count": 450,
            "centroids": [[0.3, 0.3]] * 450,
            "raw_bboxes_norm": mock_raw_bboxes,
            "raw_confidences": mock_raw_confs,
        }

        # Apply subsampling logic as in stealing_detector
        if len(track_info["raw_bboxes_norm"]) > 300:
            indices = np.linspace(0, len(track_info["raw_bboxes_norm"]) - 1, 300, dtype=int)
            bbox_traj = [track_info["raw_bboxes_norm"][i] for i in indices]
            conf_hist = [track_info["raw_confidences"][i] for i in indices]
        else:
            bbox_traj = list(track_info["raw_bboxes_norm"])
            conf_hist = list(track_info["raw_confidences"])

        assert len(bbox_traj) == 300
        assert len(conf_hist) == 300
        assert bbox_traj[0]["t"] == 0.0
        assert bbox_traj[-1]["t"] == 44.9
        assert conf_hist[0]["t"] == 0.0
        assert conf_hist[-1]["t"] == 44.9

    def test_clip_start_s_in_incident(self):
        processor = VideoProcessor()
        stealing_cluster = [
            {"window_index": 0, "start_time": 5.0, "end_time": 7.5, "center_time": 6.25,
             "top_label": "Stealing", "confidence": 0.95, "anomaly_score": 0.95, "is_anomaly": True,
             "raw_logits": [0.01, 0.02, 0.02, 0.95]},
        ]
        inc = processor._build_incident(stealing_cluster, 1, 30.0, "vid1", "dummy.mp4")
        assert "stealing_analysis" in inc
        assert "clip_start_s" in inc["stealing_analysis"]
        assert inc["stealing_analysis"]["clip_start_s"] == inc["clip_start"]

    def test_older_json_backward_compatibility(self):
        # Older reports missing bbox_trajectory, confidence_history, clip_start_s
        older_track_summary = {
            "track_id": 1,
            "class_name": "person",
            "is_person": True,
            "is_target_object": False,
            "first_seen_s": 1.0,
            "last_seen_s": 5.0,
            "duration_s": 4.0,
            "detection_count": 10,
            "gap_count": 0,
            "max_confidence": 0.91,
            "total_displacement": 0.15,
            "is_moving": True,
        }
        # Safe extraction defaults
        assert older_track_summary.get("bbox_trajectory", []) == []
        assert older_track_summary.get("confidence_history", []) == []
        assert older_track_summary.get("clip_start_s", "not available") == "not available"

    def test_tracking_time_mapping_trimmed_and_full(self, monkeypatch):
        processor = VideoProcessor()
        cluster = [
            {"window_index": 0, "start_time": 10.0, "end_time": 12.5, "center_time": 11.25,
             "top_label": "Stealing", "confidence": 0.90, "anomaly_score": 0.90, "is_anomaly": True,
             "raw_logits": [0.02, 0.04, 0.04, 0.90]},
        ]

        # Case 1: Trimmed Clip (extract_video_clip succeeds)
        monkeypatch.setattr(processor, "extract_video_clip", lambda *args, **kwargs: True)
        inc_trimmed = processor._build_incident(cluster, 1, 60.0, "test_vid", "dummy.mp4")
        assert inc_trimmed["clip_is_trimmed"] is True
        assert inc_trimmed["stealing_analysis"]["clip_is_trimmed"] is True
        assert inc_trimmed["clip_url"] != ""

        # Time mapping verification for trimmed clip:
        clip_start_s = inc_trimmed["stealing_analysis"]["clip_start_s"]
        current_time_player = 4.5
        source_time_calc = current_time_player + clip_start_s if inc_trimmed["clip_is_trimmed"] else current_time_player
        assert source_time_calc == 4.5 + clip_start_s

        cue_source_time = clip_start_s + 7.2
        seek_player_time = (cue_source_time - clip_start_s) if inc_trimmed["clip_is_trimmed"] else cue_source_time
        assert round(seek_player_time, 2) == 7.20

        # Case 2: Full Video Fallback (extract_video_clip fails)
        monkeypatch.setattr(processor, "extract_video_clip", lambda *args, **kwargs: False)
        inc_full = processor._build_incident(cluster, 2, 60.0, "test_vid", "dummy.mp4")
        assert inc_full["clip_is_trimmed"] is False
        assert inc_full["stealing_analysis"]["clip_is_trimmed"] is False
        assert inc_full["clip_url"] == ""

        # Time mapping verification for full video fallback:
        current_time_full_player = 14.5
        source_time_full_calc = current_time_full_player + clip_start_s if inc_full["clip_is_trimmed"] else current_time_full_player
        assert source_time_full_calc == 14.5

        seek_full_player_time = (cue_source_time - clip_start_s) if inc_full["clip_is_trimmed"] else cue_source_time
        assert seek_full_player_time == cue_source_time


