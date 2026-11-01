import os
import sys
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import DEFAULT_WINDOW_SECONDS, DEFAULT_STRIDE_SECONDS, ANOMALY_THRESHOLD
from app.core.model import VideoAnomalyDetector, NORMAL_NAME
from app.core.video_processor import VideoProcessor
from app.main import app


class TestModelCorrectnessAndValidation:
    @classmethod
    def setup_class(cls):
        cls.detector = VideoAnomalyDetector.get_instance()

    def test_singleton_instance(self):
        second = VideoAnomalyDetector.get_instance()
        assert self.detector is second, "VideoAnomalyDetector must be a thread-safe singleton"

    def test_probe_architecture_and_classes(self):
        assert self.detector.is_loaded, f"Detector failed to load: {self.detector.loading_error}"
        assert self.detector.engine_mode == "probe"
        assert self.detector.class_names == ["Assault", "Fighting", "Normal", "Stealing"]
        assert len(self.detector.class_names) == 4
        assert hasattr(self.detector, "anomaly_threshold")
        assert self.detector.anomaly_threshold == ANOMALY_THRESHOLD

    def test_single_window_prediction_format(self):
        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        pred = self.detector.predict_clip(dummy_frames)

        assert "top_label" in pred
        assert "final_prediction" in pred
        assert "confidence" in pred
        assert "anomaly_score" in pred
        assert "is_anomaly" in pred
        assert "severity" in pred
        assert "decision_reason" in pred
        assert "raw_scores" in pred
        assert "class_probabilities" in pred
        assert "top_3" in pred
        assert "threshold" in pred

        probs = pred["class_probabilities"]
        assert len(probs) == 4
        assert set(probs.keys()) == {"Assault", "Fighting", "Normal", "Stealing"}
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.05, f"Probabilities should sum to ~1.0, got {prob_sum}"

    def test_empty_or_invalid_model_inputs(self):
        with pytest.raises(ValueError, match="empty frames list"):
            self.detector.predict_clip([])

        with pytest.raises(TypeError, match="expected list or tuple"):
            self.detector.predict_clip("invalid_type_string")  # type: ignore

        with pytest.raises(ValueError, match="no windows"):
            self.detector.predict_windows([])

        with pytest.raises(ValueError, match="empty"):
            self.detector.predict_windows([[]])


class TestDetectionDecisionPolicy:
    @classmethod
    def setup_class(cls):
        cls.detector = VideoAnomalyDetector.get_instance()

    def test_high_normal_probability(self, monkeypatch):
        # Mock probe output: 95% Normal
        mock_probs = np.array([[0.01, 0.02, 0.95, 0.02]])
        monkeypatch.setattr(self.detector.probe, "predict_proba", lambda feats: mock_probs)
        monkeypatch.setattr(self.detector, "_window_features", lambda w: np.zeros((1, 1024)))

        dummy = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        pred = self.detector.predict_clip(dummy)

        assert pred["is_anomaly"] is False
        assert pred["top_label"] == "Normal"
        assert pred["final_prediction"] == "Normal"
        assert pred["confidence"] == 0.95
        assert pred["anomaly_score"] == 0.05
        assert "Normal:" in pred["decision_reason"]

    def test_borderline_normal_score(self, monkeypatch):
        # Normal is 0.52, Assault is 0.48. Anomaly score = 0.48 < 0.50 threshold.
        mock_probs = np.array([[0.48, 0.0, 0.52, 0.0]])
        monkeypatch.setattr(self.detector.probe, "predict_proba", lambda feats: mock_probs)
        monkeypatch.setattr(self.detector, "_window_features", lambda w: np.zeros((1, 1024)))

        self.detector.anomaly_threshold = 0.50
        dummy = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        pred = self.detector.predict_clip(dummy)

        assert pred["is_anomaly"] is False
        assert pred["top_label"] == "Normal"
        assert pred["confidence"] == 0.52
        assert pred["anomaly_score"] == 0.48

    def test_borderline_anomalous_score(self, monkeypatch):
        # Normal is 0.48, Assault is 0.52. Anomaly score = 0.52 >= 0.50 threshold.
        mock_probs = np.array([[0.52, 0.0, 0.48, 0.0]])
        monkeypatch.setattr(self.detector.probe, "predict_proba", lambda feats: mock_probs)
        monkeypatch.setattr(self.detector, "_window_features", lambda w: np.zeros((1, 1024)))

        self.detector.anomaly_threshold = 0.50
        dummy = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        pred = self.detector.predict_clip(dummy)

        assert pred["is_anomaly"] is True
        assert pred["top_label"] == "Assault"
        assert pred["confidence"] == 0.52
        assert pred["anomaly_score"] == 0.52

    def test_strong_anomalous_prediction(self, monkeypatch):
        # Strong assault: Assault 0.88, Normal 0.05. Anomaly score = 0.95.
        mock_probs = np.array([[0.88, 0.05, 0.05, 0.02]])
        monkeypatch.setattr(self.detector.probe, "predict_proba", lambda feats: mock_probs)
        monkeypatch.setattr(self.detector, "_window_features", lambda w: np.zeros((1, 1024)))

        dummy = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        pred = self.detector.predict_clip(dummy)

        assert pred["is_anomaly"] is True
        assert pred["top_label"] == "Assault"
        assert pred["confidence"] == 0.88
        assert pred["anomaly_score"] == 0.95
        assert pred["severity"] == "HIGH"

    def test_configurable_threshold_sensitivity(self, monkeypatch):
        # Anomaly score is 0.45 (Normal 0.55, Stealing 0.45)
        mock_probs = np.array([[0.0, 0.0, 0.55, 0.45]])
        monkeypatch.setattr(self.detector.probe, "predict_proba", lambda feats: mock_probs)
        monkeypatch.setattr(self.detector, "_window_features", lambda w: np.zeros((1, 1024)))

        dummy = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]

        # Under sensitive threshold (0.40): flags as anomaly
        self.detector.anomaly_threshold = 0.40
        pred_sens = self.detector.predict_clip(dummy)
        assert pred_sens["is_anomaly"] is True
        assert pred_sens["top_label"] == "Stealing"

        # Under conservative threshold (0.50): flags as normal
        self.detector.anomaly_threshold = 0.50
        pred_cons = self.detector.predict_clip(dummy)
        assert pred_cons["is_anomaly"] is False
        assert pred_cons["top_label"] == "Normal"

        # Reset to default
        self.detector.anomaly_threshold = ANOMALY_THRESHOLD


class TestTemporalAggregation:
    @classmethod
    def setup_class(cls):
        detector = VideoAnomalyDetector.get_instance()
        cls.processor = VideoProcessor(detector)

    def test_gap_merging_tolerance(self):
        timeline = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Fighting", "confidence": 0.85, "anomaly_score": 0.85, "is_anomaly": True, "severity": "HIGH"},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Normal", "confidence": 0.55, "anomaly_score": 0.45, "is_anomaly": False, "severity": "NONE"},
            {"window_index": 2, "start_time": 2.0, "end_time": 4.5, "center_time": 3.25,
             "top_label": "Fighting", "confidence": 0.90, "anomaly_score": 0.90, "is_anomaly": True, "severity": "HIGH"},
            {"window_index": 3, "start_time": 3.0, "end_time": 5.5, "center_time": 4.25,
             "top_label": "Normal", "confidence": 0.95, "anomaly_score": 0.05, "is_anomaly": False, "severity": "NONE"},
            {"window_index": 4, "start_time": 4.0, "end_time": 6.5, "center_time": 5.25,
             "top_label": "Normal", "confidence": 0.98, "anomaly_score": 0.02, "is_anomaly": False, "severity": "NONE"},
        ]

        incidents = self.processor._aggregate_incidents(
            timeline=timeline,
            fps=30.0,
            video_id="test_gap_merge",
            video_path="dummy_path.mp4",
            max_gap_windows=1,
            high_conf_single_window_threshold=0.75,
        )

        assert len(incidents) == 1, f"Expected 1 merged incident, got {len(incidents)}"
        inc = incidents[0]
        assert inc["crime_type"] == "Fighting"
        assert inc["start_time"] == 0.0
        assert inc["end_time"] == 4.5
        assert inc["review_priority"] in ["P1-CRITICAL", "P1-HIGH"]
        assert "source_video_id" in inc
        assert "responsible_ai_disclaimer" in inc

    def test_preserve_high_confidence_short_event(self):
        timeline = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Normal", "confidence": 0.95, "anomaly_score": 0.05, "is_anomaly": False, "severity": "NONE"},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Assault", "confidence": 0.88, "anomaly_score": 0.88, "is_anomaly": True, "severity": "HIGH"},
            {"window_index": 2, "start_time": 2.0, "end_time": 4.5, "center_time": 3.25,
             "top_label": "Normal", "confidence": 0.96, "anomaly_score": 0.04, "is_anomaly": False, "severity": "NONE"},
            {"window_index": 3, "start_time": 3.0, "end_time": 5.5, "center_time": 4.25,
             "top_label": "Stealing", "confidence": 0.42, "anomaly_score": 0.42, "is_anomaly": True, "severity": "MEDIUM"},
            {"window_index": 4, "start_time": 4.0, "end_time": 6.5, "center_time": 5.25,
             "top_label": "Normal", "confidence": 0.97, "anomaly_score": 0.03, "is_anomaly": False, "severity": "NONE"},
        ]

        incidents = self.processor._aggregate_incidents(
            timeline=timeline,
            fps=30.0,
            video_id="test_short_event",
            video_path="dummy_path.mp4",
            max_gap_windows=1,
            high_conf_single_window_threshold=0.75,
        )

        assert len(incidents) == 1, f"Expected exactly 1 preserved incident, got {len(incidents)}"
        assert incidents[0]["crime_type"] == "Assault"
        assert incidents[0]["confidence"] == 0.88
        assert incidents[0].get("short_event_preserved") is True

    def test_different_crime_classes_in_adjacent_windows(self):
        timeline = [
            {"window_index": 0, "start_time": 0.0, "end_time": 2.5, "center_time": 1.25,
             "top_label": "Fighting", "confidence": 0.85, "anomaly_score": 0.85, "is_anomaly": True, "severity": "HIGH"},
            {"window_index": 1, "start_time": 1.0, "end_time": 3.5, "center_time": 2.25,
             "top_label": "Fighting", "confidence": 0.88, "anomaly_score": 0.88, "is_anomaly": True, "severity": "HIGH"},
            # Immediately adjacent different crime category
            {"window_index": 2, "start_time": 2.0, "end_time": 4.5, "center_time": 3.25,
             "top_label": "Stealing", "confidence": 0.90, "anomaly_score": 0.90, "is_anomaly": True, "severity": "MEDIUM"},
            {"window_index": 3, "start_time": 3.0, "end_time": 5.5, "center_time": 4.25,
             "top_label": "Stealing", "confidence": 0.92, "anomaly_score": 0.92, "is_anomaly": True, "severity": "MEDIUM"},
        ]

        incidents = self.processor._aggregate_incidents(
            timeline=timeline,
            fps=30.0,
            video_id="test_adjacent_classes",
            video_path="dummy_path.mp4",
        )

        assert len(incidents) == 2, f"Expected 2 separate incidents for distinct classes, got {len(incidents)}"
        assert incidents[0]["crime_type"] == "Fighting"
        assert incidents[0]["start_time"] == 0.0
        assert incidents[0]["end_time"] == 3.5

        assert incidents[1]["crime_type"] == "Stealing"
        assert incidents[1]["start_time"] == 2.0
        assert incidents[1]["end_time"] == 5.5


class TestAPIEndpoints:
    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["is_loaded"] is True
        assert "device" in data
        assert "engine_mode" in data

    def test_samples_endpoint(self):
        response = self.client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert len(data["samples"]) >= 3

    def test_job_status_not_found(self):
        response = self.client.get("/api/status/job_nonexistent_12345")
        assert response.status_code == 404
