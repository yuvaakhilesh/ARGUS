import os
import sys
import tempfile
import pytest
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.video_processor import VideoProcessor
from app.main import app

client = TestClient(app)


def create_synthetic_video(frames: list, fps: float = 20.0) -> str:
    """Helper to write synthetic frames to a temporary MP4 file."""
    tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tfile.close()
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return tfile.name


class TestUninformativeVideoHandling:
    @classmethod
    def setup_class(cls):
        cls.processor = VideoProcessor()

    # -------------------------------------------------------------------------
    # Scenario 1: A completely black one-second video
    # -------------------------------------------------------------------------
    def test_completely_black_one_second_video(self):
        # 1 second of completely black frames (20 fps = 20 frames)
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(20)]
        video_path = create_synthetic_video(frames, fps=20.0)

        try:
            metadata = cls = self.processor.get_video_metadata(video_path)
            quality = self.processor.assess_video_quality(video_path, metadata)
            assert not quality["is_assessable"], "Black video must be flagged as unassessable"
            assert quality["reason"] == "completely_black_footage"

            res = self.processor.analyze_video(video_path, video_id="test_black")
            assert res is not None
            assert res.get("is_unassessable") is True, "Must set is_unassessable=True"
            assert "Unassessable" in res["verdict"], f"Expected Unassessable verdict, got {res['verdict']}"
            assert res["top_class"] == "Unassessable", f"Expected top_class Unassessable, got {res['top_class']}"
            assert res["severity"] == "NONE", f"Expected severity NONE, got {res['severity']}"
            assert res["overall_threat"] == "NONE"
            assert res["confidence"] == 0.0
            assert len(res["incidents"]) == 0, f"Must have 0 incidents, got {len(res['incidents'])}"
            assert len(res["clips"]) == 0, f"Must have 0 clips, got {len(res['clips'])}"
            assert "quality_warning" in res and res["quality_warning"] is not None
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    # -------------------------------------------------------------------------
    # Scenario 2: A blank or near-uniform video
    # -------------------------------------------------------------------------
    def test_blank_or_near_uniform_video(self):
        # 1 second of solid uniform gray (pixel intensity 128)
        frames = [np.full((240, 320, 3), 128, dtype=np.uint8) for _ in range(20)]
        video_path = create_synthetic_video(frames, fps=20.0)

        try:
            metadata = self.processor.get_video_metadata(video_path)
            quality = self.processor.assess_video_quality(video_path, metadata)
            assert not quality["is_assessable"], "Uniform gray video must be flagged as unassessable"
            assert quality["reason"] == "near_uniform_blank_footage"

            res = self.processor.analyze_video(video_path, video_id="test_uniform")
            assert res is not None
            assert res.get("is_unassessable") is True
            assert res["severity"] == "NONE"
            assert len(res["incidents"]) == 0
            assert len(res["clips"]) == 0
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    # -------------------------------------------------------------------------
    # Scenario 3: A dark but valid surveillance scene
    # -------------------------------------------------------------------------
    def test_dark_but_valid_surveillance_scene(self):
        # Nighttime camera scene: low ambient light (mean ~12) + realistic camera sensor noise + dim edge structure
        np.random.seed(123)
        frames = []
        for _ in range(20):
            noise = np.random.normal(0, 3.5, (240, 320, 3))
            base_scene = np.clip(12 + noise, 0, 255).astype(np.uint8)
            # Add dim doorway and street lamp halo
            base_scene[80:200, 60:120] = np.clip(28 + noise[80:200, 60:120], 0, 255).astype(np.uint8)
            base_scene[30:70, 200:260] = np.clip(35 + noise[30:70, 200:260], 0, 255).astype(np.uint8)
            frames.append(base_scene)
        video_path = create_synthetic_video(frames, fps=20.0)

        try:
            metadata = self.processor.get_video_metadata(video_path)
            quality = self.processor.assess_video_quality(video_path, metadata)
            assert quality["is_assessable"], f"Dark surveillance scene must NOT be rejected! Quality result: {quality}"

            # Video analysis must run through without being blocked by quality validator
            res = self.processor.analyze_video(video_path, video_id="test_dark_scene")
            assert res is not None
            assert res.get("is_unassessable") is not True, "Dark surveillance scene should be assessed"
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    # -------------------------------------------------------------------------
    # Scenario 4: A genuine fighting clip
    # -------------------------------------------------------------------------
    def test_genuine_fighting_clip_preservation(self):
        fighting_path = "test_data/Fighting/Fighting008_x264.mp4"
        assert os.path.exists(fighting_path), f"Test file {fighting_path} not found"

        metadata = self.processor.get_video_metadata(fighting_path)
        quality = self.processor.assess_video_quality(fighting_path, metadata)
        assert quality["is_assessable"], "Real fighting clip must pass quality assessment"

        res = self.processor.analyze_video(fighting_path, video_id="test_fighting")
        assert res is not None
        assert res.get("is_unassessable") is not True
        assert res["verdict"] == "Anomaly detected"
        assert res["top_class"] == "Fighting"
        assert res["severity"] in ["HIGH", "CRITICAL"]
        assert len(res["incidents"]) >= 1

    # -------------------------------------------------------------------------
    # Scenario 5: A normal clip
    # -------------------------------------------------------------------------
    def test_normal_clip_preservation(self):
        normal_path = "test_data/Normal/Normal_Videos_247_x264.mp4"
        assert os.path.exists(normal_path), f"Test file {normal_path} not found"

        metadata = self.processor.get_video_metadata(normal_path)
        quality = self.processor.assess_video_quality(normal_path, metadata)
        assert quality["is_assessable"], "Normal surveillance clip must pass quality assessment"

        res = self.processor.analyze_video(normal_path, video_id="test_normal")
        assert res is not None
        assert res.get("is_unassessable") is not True
        assert res["top_class"] in ["Normal Videos", "Normal"]

    # -------------------------------------------------------------------------
    # Scenario 6: A short but valid video
    # -------------------------------------------------------------------------
    def test_short_but_valid_video_and_timestamps(self):
        # Extract a short 1.0 second valid slice (20 frames) from real video
        cap = cv2.VideoCapture("test_data/Fighting/Fighting008_x264.mp4")
        frames = []
        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        assert len(frames) == 20

        video_path = create_synthetic_video(frames, fps=20.0)
        try:
            metadata = self.processor.get_video_metadata(video_path)
            assert metadata["duration"] >= 0.9, f"Expected ~1.0s, got {metadata['duration']}"
            quality = self.processor.assess_video_quality(video_path, metadata)
            assert quality["is_assessable"], "Short valid video with real scene activity must pass quality assessment"

            res = self.processor.analyze_video(video_path, video_id="test_short_valid")
            assert res is not None
            assert res.get("is_unassessable") is not True

            # Check interval and timestamp formatting logic
            if res["incidents"]:
                for inc in res["incidents"]:
                    assert inc["duration"] >= 0.1, "Duration must be positive"
                    # Description must not have '00:00 and 00:00' if duration > 0
                    desc = inc["description"]
                    assert "00:00 and 00:00" not in desc, f"Spurious identical timestamps in description: {desc}"
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    # -------------------------------------------------------------------------
    # API Integration: POST /upload with a completely black video
    # -------------------------------------------------------------------------
    def test_api_upload_black_video_returns_unassessable(self):
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(20)]
        video_path = create_synthetic_video(frames, fps=20.0)

        try:
            with open(video_path, "rb") as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test_black_api.mp4", f, "video/mp4")},
                )
            assert response.status_code == 200, f"Upload failed: {response.text}"
            upload_data = response.json()
            assert upload_data.get("success") is True
            job_id = upload_data.get("job_id")
            assert job_id is not None

            # Poll for job completion
            import time
            final_job = None
            for _ in range(30):
                status_res = client.get(f"/status/{job_id}")
                assert status_res.status_code == 200
                st_data = status_res.json()
                if st_data.get("status") in ["done", "completed", "error", "failed"]:
                    final_job = st_data
                    break
                time.sleep(0.1)

            assert final_job is not None, "Job timed out"
            assert final_job.get("status") in ["done", "completed"], f"Job failed: {final_job.get('error')}"
            results = final_job.get("results", {})
            assert results.get("is_unassessable") is True
            assert "Unassessable" in results.get("verdict", "")
            assert results.get("severity") == "NONE"
            assert len(results.get("incidents", [])) == 0
            assert len(results.get("clips", [])) == 0
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
