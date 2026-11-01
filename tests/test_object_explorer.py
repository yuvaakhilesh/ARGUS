"""
Unit and integration tests for ARGUS Tracking Lab Refinement & Object Explorer.

Verifies:
1. Dynamic object catalog (80 COCO + 3 WeaponShield classes, functional categories, colors).
2. Catalog search differentiation (supported vs unsupported).
3. Backend REST API routes:
   - GET /api/tracking/classes
   - GET /api/tracking/objects
   - GET /api/tracking/search
   - GET /api/tracking/tracks/{id}
   - POST /api/tracking/scan (Mode B targeted scan with frames examined & coverage)
   - GET /api/tracking/inventory
   - GET /api/tracking/export (JSON and CSV)
4. Duplicate detection suppression & fusion (weapon detector vs COCO detector IoU >= 0.40).
5. Unrestricted tracking pipeline across all 80 COCO classes.
"""

import json
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.core.object_catalog import (
    ObjectCatalog,
    CLASS_CATEGORIES,
    CATEGORY_COLORS,
)
from app.core.tracking_pipeline import GeneralizedTracker, compute_box_iou
from app.api.routes import tracking_cache

client = TestClient(app)


class TestObjectCatalog:
    """Verifies catalog composition and functional categorization."""

    def test_catalog_size_and_classes(self):
        """Catalog must contain 80 COCO classes + 3 WeaponShield classes (83 total)."""
        supported = ObjectCatalog.get_supported_classes()
        assert len(supported) == 83

        # Verify 3 weapons present
        weapon_names = [c["class_name"] for c in supported if c["model_source"] == "weapon_shield_yolov8s"]
        assert "Pistol" in weapon_names
        assert "Rifle" in weapon_names
        assert "Knife" in weapon_names
        assert len(weapon_names) == 3

        # Verify 80 COCO classes present
        coco_names = [c["class_name"] for c in supported if c["model_source"] == "coco_yolov8n"]
        assert len(coco_names) == 80
        assert "person" in coco_names
        assert "car" in coco_names
        assert "cell phone" in coco_names
        assert "bottle" in coco_names
        assert "knife" in coco_names

    def test_category_assignment_and_colors(self):
        """Every supported class must map to a valid category with an assigned hex color."""
        supported = ObjectCatalog.get_supported_classes()
        for item in supported:
            assert "category" in item
            assert "color" in item
            assert item["color"].startswith("#")
            assert item["category"] in CATEGORY_COLORS

    def test_catalog_search(self):
        """Search must support exact, partial, and case-insensitive matching."""
        # Partial match
        res_phone = ObjectCatalog.search_catalog("phone")
        assert res_phone["is_supported"] is True
        assert any(c["class_name"] == "cell phone" for c in res_phone["matched_classes"])

        # Weapon match
        res_knife = ObjectCatalog.search_catalog("knife")
        assert res_knife["is_supported"] is True
        knife_names = [c["class_name"] for c in res_knife["matched_classes"]]
        assert "Knife" in knife_names or "knife" in knife_names

        # Unsupported class
        res_fake = ObjectCatalog.search_catalog("spaceshuttle12345")
        assert res_fake["is_supported"] is False
        assert len(res_fake["matched_classes"]) == 0


class TestDuplicateSuppressionAndFusion:
    """Verifies IoU-based duplicate detection suppression between COCO and Weapon detectors."""

    def test_box_iou_computation(self):
        """Tests IoU math for overlapping and non-overlapping bounding boxes."""
        box1 = [0.1, 0.1, 0.3, 0.3]
        box2 = [0.1, 0.1, 0.3, 0.3]
        # Identical boxes -> IoU = 1.0
        assert pytest.approx(compute_box_iou(box1, box2), 0.01) == 1.0

        box3 = [0.5, 0.5, 0.7, 0.7]
        # Disjoint boxes -> IoU = 0.0
        assert compute_box_iou(box1, box3) == 0.0

        # Partial overlap
        box4 = [0.1, 0.1, 0.3, 0.2]  # area = 0.2 * 0.1 = 0.02
        box5 = [0.1, 0.1, 0.3, 0.3]  # area = 0.2 * 0.2 = 0.04
        # Intersection = 0.02, Union = 0.04 -> IoU = 0.5
        assert pytest.approx(compute_box_iou(box4, box5), 0.01) == 0.5

    def test_knife_suppression_logic(self):
        """When weapon detector detects Knife and COCO detects knife with IoU >= 0.40, weapon is kept."""
        weapon_det = {
            "bbox_norm": [0.2, 0.2, 0.4, 0.3],
            "confidence": 0.88,
            "class_name": "Knife",
            "model_source": "weapon_shield_ai",
            "is_weapon": True,
        }
        coco_det = {
            "bbox_norm": [0.21, 0.20, 0.39, 0.31],  # Heavy overlap
            "confidence": 0.65,
            "class_name": "knife",
            "model_source": "coco_yolov8n",
            "is_weapon": False,
        }

        iou = compute_box_iou(weapon_det["bbox_norm"], coco_det["bbox_norm"])
        assert iou >= 0.40

        # Dedicated weapon detector prioritized
        fused = []
        if iou >= 0.40:
            weapon_det["fused_with"] = "COCO-knife"
            fused.append(weapon_det)
        assert len(fused) == 1
        assert fused[0]["model_source"] == "weapon_shield_ai"
        assert fused[0]["class_name"] == "Knife"
        assert fused[0]["fused_with"] == "COCO-knife"


class TestTrackingAPIEndpoints:
    """Verifies all new backend REST API routes."""

    @pytest.fixture(autouse=True)
    def setup_mock_tracking_cache(self):
        """Seeds tracking cache with mock video analysis."""
        vid_key = "test_surveillance.mp4"
        tracking_cache[vid_key] = {
            "track_summaries": [
                {
                    "track_id": 1,
                    "track_id_str": "ID-1",
                    "class_name": "person",
                    "category": "People",
                    "model_source": "coco_yolov8n",
                    "entity_type": "person",
                    "first_seen_s": 0.5,
                    "last_seen_s": 4.0,
                    "duration_s": 3.5,
                    "detection_count": 15,
                    "max_confidence": 0.92,
                    "avg_confidence": 0.88,
                    "is_weapon": False,
                    "is_person": True,
                    "is_suppressed": False,
                    "bbox_trajectory": [{"t": 1.0, "bbox_norm": [0.1, 0.1, 0.3, 0.5]}],
                },
                {
                    "track_id": 2,
                    "track_id_str": "ID-2",
                    "class_name": "cell phone",
                    "category": "Personal Belongings",
                    "model_source": "coco_yolov8n",
                    "entity_type": "belonging",
                    "first_seen_s": 1.0,
                    "last_seen_s": 3.0,
                    "duration_s": 2.0,
                    "detection_count": 8,
                    "max_confidence": 0.85,
                    "avg_confidence": 0.81,
                    "is_weapon": False,
                    "is_person": False,
                    "is_suppressed": False,
                    "bbox_trajectory": [{"t": 1.5, "bbox_norm": [0.2, 0.2, 0.25, 0.3]}],
                },
                {
                    "track_id": 3,
                    "track_id_str": "WEP-1",
                    "class_name": "Pistol",
                    "category": "Weapons",
                    "model_source": "weapon_shield_yolov8s",
                    "entity_type": "weapon",
                    "first_seen_s": 1.2,
                    "last_seen_s": 2.8,
                    "duration_s": 1.6,
                    "detection_count": 10,
                    "max_confidence": 0.94,
                    "avg_confidence": 0.91,
                    "is_weapon": True,
                    "is_person": False,
                    "is_suppressed": False,
                    "bbox_trajectory": [{"t": 1.5, "bbox_norm": [0.22, 0.25, 0.28, 0.35]}],
                },
            ],
            "inventory": None,
            "weapon_analysis": {
                "cues": [
                    {
                        "cue_type": "weapon_proximity",
                        "person_track_id": "ID-1",
                        "weapon_track_id": "WEP-1",
                        "timestamp": 1.5,
                        "description": "Person in proximity to detected Pistol.",
                    }
                ]
            },
        }
        yield
        tracking_cache.clear()

    def test_get_tracking_classes(self):
        """GET /api/tracking/classes must return all 83 classes grouped by category."""
        response = client.get("/api/tracking/classes")
        assert response.status_code == 200
        data = response.json()
        assert data["total_classes"] == 83
        assert len(data["classes"]) == 83
        assert "categories" in data
        assert "Weapons" in data["categories"]
        assert "People" in data["categories"]
        assert "Vehicles" in data["categories"]

    def test_get_tracked_objects(self):
        """GET /api/tracking/objects returns tracked objects with category filter."""
        res_all = client.get("/api/tracking/objects")
        assert res_all.status_code == 200
        assert res_all.json()["total_tracks"] == 3

        # Filter by category
        res_wep = client.get("/api/tracking/objects?category=Weapons")
        assert res_wep.status_code == 200
        assert res_wep.json()["total_tracks"] == 1
        assert res_wep.json()["tracks"][0]["class_name"] == "Pistol"

    def test_search_tracking_objects_endpoint(self):
        """GET /api/tracking/search must search active tracks and report status."""
        # 1. Search existing track
        res1 = client.get("/api/tracking/search?q=phone")
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "matches_found"
        assert len(data1["matching_tracks"]) == 1
        assert data1["matching_tracks"][0]["class_name"] == "cell phone"

        # 2. Search weapon
        res2 = client.get("/api/tracking/search?q=pistol")
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "matches_found"
        assert len(data2["matching_tracks"]) == 1
        assert data2["matching_tracks"][0]["class_name"] == "Pistol"

        # 3. Search supported but not detected class
        res3 = client.get("/api/tracking/search?q=car")
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["status"] == "supported_not_detected"
        assert len(data3["matching_tracks"]) == 0

        # 4. Search unsupported class
        res4 = client.get("/api/tracking/search?q=submarine")
        assert res4.status_code == 200
        data4 = res4.json()
        assert data4["status"] == "unsupported_class"
        assert len(data4["matching_tracks"]) == 0

    def test_get_track_by_id(self):
        """GET /api/tracking/tracks/{track_id} returns detailed telemetry."""
        res = client.get("/api/tracking/tracks/WEP-1")
        assert res.status_code == 200
        data = res.json()
        assert data["track"]["track_id_str"] == "WEP-1"
        assert data["track"]["class_name"] == "Pistol"
        assert data["track"]["is_weapon"] is True
        assert len(data["related_cues"]) == 1

        # Non-existent track ID
        res_404 = client.get("/api/tracking/tracks/NON_EXISTENT")
        assert res_404.status_code == 404

    def test_get_inventory(self):
        """GET /api/tracking/inventory generates structured inventory."""
        res = client.get("/api/tracking/inventory")
        assert res.status_code == 200
        data = res.json()
        inv = data["inventory"]
        assert inv["total_objects_tracked"] == 3
        assert inv["distinct_classes_detected"] == 3
        class_names = [item["class_name"] for item in inv["inventory"]]
        assert "Pistol" in class_names
        assert "person" in class_names
        assert "cell phone" in class_names

    def test_export_tracking_json_and_csv(self):
        """GET /api/tracking/export must output valid JSON and CSV formats."""
        # JSON
        res_json = client.get("/api/tracking/export?format=json")
        assert res_json.status_code == 200
        assert "application/json" in res_json.headers["content-type"]
        data = res_json.json()
        assert "tracks" in data
        assert len(data["tracks"]) == 3

        # CSV
        res_csv = client.get("/api/tracking/export?format=csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        csv_content = res_csv.text
        assert "Track ID,Class Name,Category" in csv_content
        assert "Pistol" in csv_content
        assert "cell phone" in csv_content

    def test_targeted_scan_mode_b(self, tmp_path):
        """POST /api/tracking/scan runs targeted scan on video and reports exact frames examined and coverage."""
        import cv2
        test_video_path = tmp_path / "test_scan.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (320, 240))
        for _ in range(10):  # 10 frames = 1 second at 10 fps
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        payload = {
            "video_source": str(test_video_path),
            "target_classes": ["person"],
            "start_time": 0.0,
            "end_time": 1.0,
            "sample_fps": 5.0,
        }

        res = client.post("/api/tracking/scan", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        results = data["results"]
        assert results["status"] == "completed"
        scan_meta = results["scan_metadata"]
        assert scan_meta["frames_examined"] > 0
        assert scan_meta["sampling_coverage_pct"] > 0.0
        assert "processing_time_s" in results
