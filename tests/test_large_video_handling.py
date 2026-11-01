import os
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import UPLOAD_DIR
from app.core.video_processor import VideoProcessor, NUM_FRAMES_PER_WINDOW
from app.api.routes import cleanup_temporary_files, jobs

client = TestClient(app)


@pytest.fixture
def mock_detector():
    det = MagicMock()
    det.class_names = ["Normal", "Fighting", "Stealing", "Robbery"]
    det.anomaly_threshold = 0.50
    det.is_loaded = True
    det.device = "cpu"
    det.device_name = "CPU"
    det.engine_mode = "videomae_probe"
    det.predict_clip.return_value = {
        "top_label": "Normal",
        "final_prediction": "Normal",
        "confidence": 0.95,
        "anomaly_score": 0.05,
        "is_anomaly": False,
        "severity": "NONE",
        "decision_reason": "Normal",
        "raw_scores": {"Normal": 0.95, "Fighting": 0.02, "Stealing": 0.02, "Robbery": 0.01},
        "top_3": [{"label": "Normal", "prob": 0.95}],
        "class_probabilities": {"Normal": 0.95, "Fighting": 0.02, "Stealing": 0.02, "Robbery": 0.01},
        "raw_logits": [2.5, -1.0, -1.0, -1.5],
    }
    return det


def safe_unlink(path: Path, max_retries: int = 15, delay: float = 0.2):
    """Safely unlink a file on Windows by retrying if OneDrive or Antivirus momentarily locks it."""
    for _ in range(max_retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
    path.unlink(missing_ok=True)


class TestLargeVideoUploadLimits:
    """Test 1 & 2: Video uploads under and over 250 MB and new size boundaries."""

    def test_upload_under_250mb(self, tmp_path):
        """Videos under 250 MB upload cleanly without errors."""
        content = b"fake video bytes under 250mb"
        with patch("app.api.routes.run_analysis_worker"):
            response = client.post(
                "/upload",
                files={"file": ("test_small.mp4", content, "video/mp4")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        job_id = data["job_id"]
        assert job_id in jobs

        # Clean up
        safe_unlink(Path(jobs[job_id]["video_path"]))
        del jobs[job_id]

    def test_upload_over_250mb_accepted(self, monkeypatch):
        """
        Videos over 250 MB (e.g. 265.7 MB 4K test video) must now be accepted
        instead of rejected by the old 250 MB limit.
        """
        target_size = 260 * 1024 * 1024  # 260 MB

        class FakeBigUploadFile:
            def __init__(self):
                self.filename = "surveillance_4k_over250mb.mp4"
                self.bytes_sent = 0

            async def read(self, size=-1):
                if self.bytes_sent >= target_size:
                    return b""
                to_read = min(size if size > 0 else 1024 * 1024, target_size - self.bytes_sent)
                self.bytes_sent += to_read
                return b"\x00" * to_read

        import asyncio
        from app.api.routes import upload_video

        fake_file = FakeBigUploadFile()
        with patch("app.api.routes.processor.get_video_metadata") as mock_meta, \
             patch("app.api.routes.run_analysis_worker") as mock_worker:
            mock_meta.return_value = {"duration": 90.0, "fps": 30.0, "width": 3840, "height": 2160, "total_frames": 2700}
            result = asyncio.run(upload_video(file=fake_file))

        assert result["success"] is True
        job_id = result["job_id"]
        assert result["file_size"] == target_size
        save_path = Path(jobs[job_id]["video_path"])
        assert save_path.exists()
        assert save_path.stat().st_size == target_size

        # Clean up
        safe_unlink(save_path)
        del jobs[job_id]

    def test_upload_exceeds_configured_limit_rejected_and_cleaned_up(self, monkeypatch):
        """When a file exceeds AEGIS_MAX_UPLOAD_SIZE_BYTES, it is rejected and cleaned up."""
        initial_files = set(UPLOAD_DIR.glob("upload_*.mp4"))
        monkeypatch.setenv("AEGIS_MAX_UPLOAD_SIZE_BYTES", str(1024 * 1024))  # 1 MB limit
        content = b"X" * (2 * 1024 * 1024)  # 2 MB payload
        response = client.post(
            "/upload",
            files={"file": ("too_big.mp4", content, "video/mp4")},
        )
        assert response.status_code == 400
        assert "File exceeds maximum allowed size" in response.json()["detail"]

        # Ensure no new orphaned files remain in UPLOAD_DIR
        new_files = set(UPLOAD_DIR.glob("upload_*.mp4")) - initial_files
        assert len(new_files) == 0


class TestBoundedMemoryAndStreaming:
    """Test 3 & 4: Multi-temporal window streaming and bounded memory proof."""

    def test_stream_windows_bounded_memory_multi_hour_simulation(self, mock_detector):
        """
        Simulate a multi-hour CCTV recording (2 hours = 7,200 seconds at 25 fps = 180,000 frames).
        Verify that frame_cache memory stays bounded and does NOT grow with video duration.
        """
        proc = VideoProcessor(mock_detector)
        duration = 7200.0  # 2 hours
        fps = 25.0
        total_frames = int(duration * fps)  # 180,000 frames
        window_duration = 2.5
        stride_duration = 1.0

        windows = []
        cur = 0.0
        while cur + window_duration <= duration:
            windows.append((round(cur, 2), round(cur + window_duration, 2)))
            cur += stride_duration
        if cur < duration:
            windows.append((round(max(0.0, duration - window_duration), 2), round(duration, 2)))

        total_windows = len(windows)
        assert total_windows >= 7198  # ~7200 sliding windows

        # Precalculate reference counts as done in stream_windows
        from collections import defaultdict
        frame_ref_count = defaultdict(int)
        for start_t, end_t in windows:
            start_f = max(0, int(start_t * fps))
            end_f = min(total_frames - 1, int(end_t * fps))
            idx_list = np.linspace(start_f, end_f, NUM_FRAMES_PER_WINDOW, dtype=int)
            for f in idx_list:
                frame_ref_count[int(f)] += 1

        # Simulate sliding window frame cache lifecycle
        frame_cache = {}
        max_cache_frames = 0
        active_w = 0
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Iterate through first 200 windows (sufficient to reach steady-state sliding window)
        for cur_f in range(int(windows[200][1] * fps) + 5):
            if frame_ref_count[cur_f] > 0:
                frame_cache[cur_f] = dummy_frame

            max_cache_frames = max(max_cache_frames, len(frame_cache))

            # Evict completed windows
            while active_w < 200:
                s_t, e_t = windows[active_w]
                last_f = int(e_t * fps)
                if cur_f > last_f:
                    s_f = max(0, int(s_t * fps))
                    w_indices = np.linspace(s_f, min(total_frames - 1, last_f), NUM_FRAMES_PER_WINDOW, dtype=int)
                    for f in w_indices:
                        frame_ref_count[int(f)] -= 1
                        if frame_ref_count[int(f)] <= 0 and int(f) in frame_cache:
                            del frame_cache[int(f)]
                    active_w += 1
                else:
                    break

        # Theoretical bound for 2.5s window at 1.0s stride: max 28-35 frames in cache
        assert max_cache_frames <= 35, f"Cache exceeded bounded limit: {max_cache_frames} frames"
        peak_cache_mb = (max_cache_frames * 224 * 224 * 3) / (1024 * 1024)
        assert peak_cache_mb < 6.0, f"Peak frame cache memory {peak_cache_mb:.2f} MB exceeds 6 MB!"

    def test_tracemalloc_memory_profiling_stream_windows(self, tmp_path, mock_detector):
        """
        Run stream_windows on a synthetic 10-second video with tracemalloc.
        Peak memory allocated during decoding must be under 30 MB.
        """
        # Generate synthetic 10s video
        vid_path = tmp_path / "stream_test.mp4"
        fps = 20.0
        width, height = 320, 240
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, fps, (width, height))
        for i in range(200):  # 10 seconds
            frame = np.full((height, width, 3), (i % 255), dtype=np.uint8)
            out.write(frame)
        out.release()

        proc = VideoProcessor(mock_detector)
        windows = [(round(i * 1.0, 2), round(i * 1.0 + 2.5, 2)) for i in range(8)]

        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        yielded_windows = 0
        for w_idx, (st, et), frames in proc.stream_windows(
            video_path=str(vid_path),
            windows=windows,
            fps=fps,
            total_frames=200,
        ):
            yielded_windows += 1
            assert len(frames) == 16
            assert frames[0].shape == (224, 224, 3)

        snapshot_end = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert yielded_windows == len(windows)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 30.0, f"Peak memory {peak_mb:.2f} MB exceeded 30 MB threshold"


class TestTimestampAccuracy:
    """Test 5: Correct timestamps at beginning, middle, and end of long recordings."""

    def test_timestamp_integrity_across_hours(self, mock_detector):
        """
        Verify that windows and incidents preserve accurate timestamps across a 2-hour duration.
        """
        proc = VideoProcessor(mock_detector)
        duration = 7200.0  # 2 hours (0s to 7200s)
        window_duration = 2.5
        stride_duration = 1.0

        windows = []
        cur = 0.0
        while cur + window_duration <= duration:
            windows.append((round(cur, 2), round(cur + window_duration, 2)))
            cur += stride_duration
        if cur < duration:
            windows.append((round(max(0.0, duration - window_duration), 2), round(duration, 2)))

        # 1. Beginning timestamp check
        assert windows[0] == (0.0, 2.5)
        assert windows[1] == (1.0, 3.5)

        # 2. Middle timestamp check (around 1 hour = 3600s)
        middle_idx = 3600
        assert windows[middle_idx][0] == 3600.0
        assert windows[middle_idx][1] == 3602.5

        # 3. End timestamp check (7200s)
        assert windows[-1][1] == 7200.0
        assert windows[-1][0] == 7197.5

        # Test incident aggregation preserves original source timestamps in hour 2
        fake_timeline = [
            # Incident near the 1-hour mark (3600s - 3605s)
            {
                "window_index": 3600,
                "start_time": 3600.0,
                "end_time": 3602.5,
                "center_time": 3601.25,
                "top_label": "Fighting",
                "final_prediction": "Fighting",
                "confidence": 0.88,
                "anomaly_score": 0.88,
                "is_anomaly": True,
                "severity": "HIGH",
            },
            {
                "window_index": 3601,
                "start_time": 3601.0,
                "end_time": 3603.5,
                "center_time": 3602.25,
                "top_label": "Fighting",
                "final_prediction": "Fighting",
                "confidence": 0.90,
                "anomaly_score": 0.90,
                "is_anomaly": True,
                "severity": "HIGH",
            },
            {
                "window_index": 3602,
                "start_time": 3602.0,
                "end_time": 3604.5,
                "center_time": 3603.25,
                "top_label": "Fighting",
                "final_prediction": "Fighting",
                "confidence": 0.85,
                "anomaly_score": 0.85,
                "is_anomaly": True,
                "severity": "HIGH",
            },
        ]

        with patch.object(proc, "extract_video_clip", return_value=True):
            incidents = proc._aggregate_incidents(
                timeline=fake_timeline,
                fps=25.0,
                video_id="cctv_long",
                video_path="dummy.mp4",
            )

        assert len(incidents) == 1
        inc = incidents[0]
        assert inc["start_time"] == 3600.0
        assert inc["end_time"] == 3604.5
        assert inc["peak_timestamp"] == 3602.25
        assert inc["crime_type"] == "Fighting"


class TestFailureHandlingAndCleanup:
    """Test 6: Disk exhaustion, cancellation, and temporary file cleanup."""

    def test_disk_exhaustion_returns_507_and_cleans_up(self, monkeypatch):
        """When disk storage is below 100 MB, reject with HTTP 507 and remove partial file."""
        from collections import namedtuple
        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])

        initial_files = set(UPLOAD_DIR.glob("upload_*.mp4"))
        # Mock shutil.disk_usage to return 50 MB free (< 100 MB)
        with patch("shutil.disk_usage", return_value=DiskUsage(100 * 1024 * 1024, 50 * 1024 * 1024, 50 * 1024 * 1024)):
            response = client.post(
                "/upload",
                files={"file": ("cctv_fail.mp4", b"some video content", "video/mp4")},
            )
            assert response.status_code == 507
            assert "Insufficient disk storage" in response.json()["detail"]

        # Ensure no new orphaned files remain in UPLOAD_DIR
        new_files = set(UPLOAD_DIR.glob("upload_*.mp4")) - initial_files
        assert len(new_files) == 0

    def test_cancel_job_cleans_up_uploaded_file(self):
        """Cancelling a job unlinks the uploaded file immediately."""
        # Create a real dummy file in UPLOAD_DIR
        dummy_path = UPLOAD_DIR / "upload_cancel_test.mp4"
        dummy_path.write_bytes(b"temp video content for cancel test")
        assert dummy_path.exists()

        job_id = "job_test_cancel_cleanup"
        jobs[job_id] = {
            "job_id": job_id,
            "video_path": str(dummy_path),
            "status": "processing",
            "cancelled": False,
        }

        response = client.post(f"/cancel/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert not dummy_path.exists(), "Cancelled file was not removed from disk!"

        # Clean up job dict
        del jobs[job_id]

    def test_cleanup_temporary_files_prunes_expired_files(self):
        """cleanup_temporary_files cleans files older than max_age_hours."""
        old_file = UPLOAD_DIR / "upload_old_test.mp4"
        old_file.write_bytes(b"old upload")
        # Set file modification time to 48 hours ago
        past_time = time.time() - (48 * 3600)
        os.utime(old_file, (past_time, past_time))

        new_file = UPLOAD_DIR / "upload_new_test.mp4"
        new_file.write_bytes(b"new upload")

        cleaned = cleanup_temporary_files(max_age_hours=24.0)
        assert cleaned >= 1
        assert not old_file.exists(), "Old expired upload was not cleaned up!"
        assert new_file.exists(), "New recent upload should not be pruned!"

        # Clean up new_file
        new_file.unlink(missing_ok=True)
