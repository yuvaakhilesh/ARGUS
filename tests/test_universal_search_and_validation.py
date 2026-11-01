"""
Comprehensive test suite for ARGUS Tracking Lab Refinement:
- Universal Custom Object Search & Natural Language Alias Resolution
- False-Positive Reduction & Calibrated Confidence Thresholds
- Temporal Object Exploration & Appearance Intervals
- Track Confirmation vs Uncertain Candidate Status
- Reproduction Case Verification (upload_5913f252_incident_1.mp4)
- True Positive Preservation on Weapon Fixtures
- API Endpoints: /api/tracking/resolve-query, /api/tracking/search, /api/tracking/export
"""

import os
# Ensure offline execution for HuggingFace transformers in sandbox
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from unittest.mock import MagicMock
from pathlib import Path
import json
import pytest

from app.core.config import (
    CLASS_CONFIDENCE_THRESHOLDS,
    DEFAULT_CLASS_CONFIDENCE_THRESHOLD,
    TRACK_CONFIRMATION_MIN_OBSERVATIONS,
    APPEARANCE_INTERVAL_MAX_GAP_SECONDS,
)
from app.core.object_catalog import ObjectCatalog, CLASS_ALIASES
from app.core.tracking_pipeline import (
    GeneralizedTracker,
    compute_appearance_intervals,
    evaluate_person_weapon_association,
)
from app.api.routes import (
    tracking_cache,
    resolve_query_endpoint,
    search_tracked_objects,
    export_tracking_data,
)


class TestQueryResolutionAndAliases:
    """Verifies natural language aliases and query resolution across all object categories."""

    def test_direct_class_queries(self):
        # Case insensitive exact classes
        res_person = ObjectCatalog.resolve_query("Person")
        assert res_person["is_supported"] is True
        assert "person" in res_person["canonical_classes"]

        res_car = ObjectCatalog.resolve_query("car")
        assert res_car["is_supported"] is True
        assert "car" in res_car["canonical_classes"]

        res_pistol = ObjectCatalog.resolve_query("Pistol")
        assert res_pistol["is_supported"] is True
        assert "Pistol" in res_pistol["canonical_classes"]

    def test_weapon_aliases(self):
        # Handgun, revolver, firearm -> Pistol
        res_handgun = ObjectCatalog.resolve_query("handgun")
        assert res_handgun["is_supported"] is True
        assert "Pistol" in res_handgun["canonical_classes"]
        assert "handgun" in res_handgun["matched_aliases"]

        res_revolver = ObjectCatalog.resolve_query("revolver")
        assert res_revolver["is_supported"] is True
        assert "Pistol" in res_revolver["canonical_classes"]

        # Gun -> both Pistol and Rifle
        res_gun = ObjectCatalog.resolve_query("gun")
        assert res_gun["is_supported"] is True
        assert "Pistol" in res_gun["canonical_classes"]
        assert "Rifle" in res_gun["canonical_classes"]

        # Blade, dagger, sword -> Knife
        res_blade = ObjectCatalog.resolve_query("blade")
        assert res_blade["is_supported"] is True
        assert any(c.lower() == "knife" for c in res_blade["canonical_classes"])

    def test_common_object_and_vehicle_aliases(self):
        # Mobile, phone, smartphone -> cell phone
        res_mobile = ObjectCatalog.resolve_query("mobile")
        assert res_mobile["is_supported"] is True
        assert "cell phone" in res_mobile["canonical_classes"]

        res_phone = ObjectCatalog.resolve_query("phone")
        assert res_phone["is_supported"] is True
        assert "cell phone" in res_phone["canonical_classes"]

        # Auto, automobile, sedan, vehicle -> car
        res_auto = ObjectCatalog.resolve_query("auto")
        assert res_auto["is_supported"] is True
        assert "car" in res_auto["canonical_classes"]

        # Bike, bicycle, motorcycle
        res_bike = ObjectCatalog.resolve_query("bike")
        assert res_bike["is_supported"] is True
        assert "bicycle" in res_bike["canonical_classes"]
        assert "motorcycle" in res_bike["canonical_classes"]

        # Bag -> backpack, handbag, suitcase
        res_bag = ObjectCatalog.resolve_query("bag")
        assert res_bag["is_supported"] is True
        assert "backpack" in res_bag["canonical_classes"]
        assert "handbag" in res_bag["canonical_classes"]
        assert "suitcase" in res_bag["canonical_classes"]

        # Purse -> handbag
        res_purse = ObjectCatalog.resolve_query("purse")
        assert res_purse["is_supported"] is True
        assert "handbag" in res_purse["canonical_classes"]

        # Brolly, parasol -> umbrella
        res_brolly = ObjectCatalog.resolve_query("brolly")
        assert res_brolly["is_supported"] is True
        assert "umbrella" in res_brolly["canonical_classes"]

        # Dog aliases
        res_pup = ObjectCatalog.resolve_query("pup")
        assert res_pup["is_supported"] is True
        assert "dog" in res_pup["canonical_classes"]

    def test_unsupported_queries(self):
        res_fake = ObjectCatalog.resolve_query("alien_laser_cannon")
        assert res_fake["is_supported"] is False
        assert len(res_fake["canonical_classes"]) == 0
        assert "not supported" in res_fake["explanation"] or "not recognized" in res_fake["explanation"]

    def test_calibrated_thresholds_lookup(self):
        # Pistol threshold is calibrated higher (0.50) to suppress car reflections
        assert ObjectCatalog.get_class_threshold("Pistol") == 0.50
        assert ObjectCatalog.get_class_threshold("Rifle") == 0.45
        assert ObjectCatalog.get_class_threshold("Knife") == 0.40
        assert ObjectCatalog.get_class_threshold("clock") == 0.50
        assert ObjectCatalog.get_class_threshold("umbrella") == 0.45
        assert ObjectCatalog.get_class_threshold("person") == 0.35
        # Default fallback
        assert ObjectCatalog.get_class_threshold("unknown_object") == DEFAULT_CLASS_CONFIDENCE_THRESHOLD


class TestAppearanceIntervals:
    """Verifies temporal appearance interval calculation across gap thresholds."""

    def test_continuous_observations_single_interval(self):
        # Trajectory with continuous frames (gap 0.2s <= 1.5s)
        traj = [
            {"t": 1.0, "confidence": 0.8},
            {"t": 1.2, "confidence": 0.85},
            {"t": 1.4, "confidence": 0.9},
            {"t": 1.6, "confidence": 0.88},
            {"t": 1.8, "confidence": 0.82},
        ]
        intervals = compute_appearance_intervals(traj, max_gap_seconds=1.5)
        assert len(intervals) == 1
        inv = intervals[0]
        assert inv["start_s"] == 1.0
        assert inv["end_s"] == 1.8
        assert round(inv["duration_s"], 2) == 0.80
        assert inv["observation_count"] == 5
        assert inv["is_single_frame"] is False

    def test_separated_observations_multiple_intervals(self):
        # Trajectory with a 4.0 second gap between t=2.0 and t=6.0 (> 1.5s)
        traj = [
            {"t": 1.0, "confidence": 0.8},
            {"t": 1.5, "confidence": 0.82},
            {"t": 2.0, "confidence": 0.85},
            # Gap of 4.0s
            {"t": 6.0, "confidence": 0.9},
            {"t": 6.5, "confidence": 0.92},
        ]
        intervals = compute_appearance_intervals(traj, max_gap_seconds=1.5)
        assert len(intervals) == 2
        
        inv1 = intervals[0]
        assert inv1["start_s"] == 1.0
        assert inv1["end_s"] == 2.0
        assert inv1["observation_count"] == 3

        inv2 = intervals[1]
        assert inv2["start_s"] == 6.0
        assert inv2["end_s"] == 6.5
        assert inv2["observation_count"] == 2

    def test_single_frame_observation(self):
        traj = [{"t": 3.45, "confidence": 0.72}]
        intervals = compute_appearance_intervals(traj, max_gap_seconds=1.5)
        assert len(intervals) == 1
        inv = intervals[0]
        assert inv["start_s"] == 3.45
        assert inv["end_s"] == 3.45
        assert inv["duration_s"] == 0.0
        assert inv["observation_count"] == 1
        assert inv["is_single_frame"] is True

    def test_empty_trajectory(self):
        intervals = compute_appearance_intervals([])
        assert intervals == []


class TestFalsePositiveEliminationAndReproduction:
    """Verifies that false-positive Pistol detections on the red car clip are eliminated."""

    def test_reproduction_clip_suppresses_false_pistol_tracks(self):
        reproduction_clip = Path("app/static/clips/upload_5913f252_incident_1.mp4")
        if not reproduction_clip.exists():
            pytest.skip("Reproduction clip upload_5913f252_incident_1.mp4 not found in clips directory")

        tracker = GeneralizedTracker(enable_weapon_detection=True)
        res = tracker.track_video(
            video_path=str(reproduction_clip),
            start_s=0.0,
            end_s=8.48,
            target_fps=5.0,
        )

        track_summaries = res.get("track_summaries", [])
        pistol_tracks = [t for t in track_summaries if t["class_name"] == "Pistol"]

        # Confirm that NO confirmed pistol tracks exist
        confirmed_pistols = [t for t in pistol_tracks if t.get("detection_status") == "confirmed_track"]
        assert len(confirmed_pistols) == 0, f"Expected 0 confirmed pistol tracks, got: {confirmed_pistols}"

        # If any isolated candidate was detected, verify it is marked as uncertain_candidate
        for pt in pistol_tracks:
            assert pt["detection_status"] == "uncertain_candidate"
            assert pt["detections_count"] < TRACK_CONFIRMATION_MIN_OBSERVATIONS

    def test_person_weapon_association_does_not_escalate_on_uncertain_candidate(self):
        person_tracks = [{"track_id": 1, "class_name": "person", "is_person": True, "bbox_trajectory": [{"t": 1.0, "bbox_norm": [0.1, 0.1, 0.5, 0.5]}]}]
        
        # Uncertain candidate weapon (single frame)
        uncertain_weapon = [{
            "track_id": 99,
            "class_name": "Pistol",
            "is_weapon": True,
            "detection_status": "uncertain_candidate",
            "detections_count": 1,
            "max_confidence": 0.46,
            "bbox_trajectory": [{"t": 1.0, "bbox_norm": [0.2, 0.2, 0.3, 0.3]}],
        }]

        assoc = evaluate_person_weapon_association(person_tracks, uncertain_weapon)
        assert assoc["status"] == "WEAPON_DETECTION_UNCERTAIN"
        assert assoc["review_recommendation"] == "STANDARD_REVIEW"
        assert assoc["confirmed_weapons"] == []  # Not elevated to confirmed threat


class TestTruePositiveRetention:
    """Verifies that genuine weapons and persons in test fixtures remain reliably confirmed."""

    def test_fixture_pistol_confirmed(self):
        fixture_path = Path("tests/fixtures/tracking/fixture_pistol.mp4")
        if not fixture_path.exists():
            pytest.skip("fixture_pistol.mp4 not found")

        tracker = GeneralizedTracker(enable_weapon_detection=True)
        res = tracker.track_video(
            video_path=str(fixture_path),
            start_s=0.0,
            end_s=3.0,
            target_fps=5.0,
        )

        pistol_tracks = [t for t in res.get("track_summaries", []) if t["class_name"] == "Pistol"]
        assert len(pistol_tracks) >= 1
        confirmed_pistols = [t for t in pistol_tracks if t.get("detection_status") == "confirmed_track"]
        assert len(confirmed_pistols) >= 1
        best = confirmed_pistols[0]
        assert best["max_confidence"] >= 0.70
        assert len(best["appearance_intervals"]) >= 1

    def test_fixture_knife_confirmed(self):
        fixture_path = Path("tests/fixtures/tracking/fixture_knife.mp4")
        if not fixture_path.exists():
            pytest.skip("fixture_knife.mp4 not found")

        tracker = GeneralizedTracker(enable_weapon_detection=True)
        res = tracker.track_video(
            video_path=str(fixture_path),
            start_s=0.0,
            end_s=3.0,
            target_fps=5.0,
        )

        knife_tracks = [t for t in res.get("track_summaries", []) if t["class_name"] == "Knife"]
        assert len(knife_tracks) >= 1
        confirmed_knives = [t for t in knife_tracks if t.get("detection_status") == "confirmed_track"]
        assert len(confirmed_knives) >= 1
        assert confirmed_knives[0]["max_confidence"] >= 0.50

    def test_fixture_normal_no_weapons(self):
        fixture_path = Path("tests/fixtures/tracking/fixture_normal_no_weapon.mp4")
        if not fixture_path.exists():
            pytest.skip("fixture_normal_no_weapon.mp4 not found")

        tracker = GeneralizedTracker(enable_weapon_detection=True)
        res = tracker.track_video(
            video_path=str(fixture_path),
            start_s=0.0,
            end_s=2.0,
            target_fps=5.0,
        )

        weapon_tracks = [t for t in res.get("track_summaries", []) if t["is_weapon"]]
        assert len(weapon_tracks) == 0


class TestAPIEndpoints:
    """Verifies FastAPI endpoints for query resolution, search, and export."""

    def test_resolve_query_endpoint(self):
        data = resolve_query_endpoint(q="handgun")
        assert data["success"] is True
        assert data["is_supported"] is True
        assert "Pistol" in data["canonical_classes"]
        assert "handgun" in data["matched_aliases"]

        # Unsupported class
        data_alien = resolve_query_endpoint(q="alien_raygun")
        assert data_alien["success"] is True
        assert data_alien["is_supported"] is False
        assert len(data_alien["canonical_classes"]) == 0

    def test_search_and_filter_endpoint_with_status_filter(self):
        # Populate test cache
        tracking_cache["test_job_123"] = {
            "track_summaries": [
                {
                    "track_id": 1,
                    "track_id_str": "ID-1",
                    "class_name": "person",
                    "category": "People",
                    "is_person": True,
                    "is_weapon": False,
                    "model_source": "coco_yolov8n",
                    "detection_status": "confirmed_track",
                    "detections_count": 5,
                    "first_seen_s": 1.0,
                    "last_seen_s": 2.5,
                    "duration_s": 1.5,
                    "avg_confidence": 0.85,
                    "max_confidence": 0.92,
                    "calibrated_threshold": 0.35,
                    "appearance_intervals": [{"start_s": 1.0, "end_s": 2.5, "display_text": "1.0s – 2.5s (1.5s)"}],
                    "bbox_trajectory": [{"t": 1.0, "bbox_norm": [0.1, 0.1, 0.2, 0.2]}, {"t": 2.5, "bbox_norm": [0.15, 0.15, 0.25, 0.25]}],
                    "is_suppressed": False,
                },
                {
                    "track_id": 2,
                    "track_id_str": "WEP-1",
                    "class_name": "Pistol",
                    "category": "Weapons",
                    "is_person": False,
                    "is_weapon": True,
                    "model_source": "weapon_shield_ai",
                    "detection_status": "uncertain_candidate",
                    "detections_count": 1,
                    "first_seen_s": 3.2,
                    "last_seen_s": 3.2,
                    "duration_s": 0.0,
                    "avg_confidence": 0.46,
                    "max_confidence": 0.46,
                    "calibrated_threshold": 0.50,
                    "candidate_reason": "Single-frame detection below confidence threshold",
                    "appearance_intervals": [{"start_s": 3.2, "end_s": 3.2, "display_text": "3.2s"}],
                    "bbox_trajectory": [{"t": 3.2, "bbox_norm": [0.3, 0.3, 0.35, 0.35]}],
                    "is_suppressed": False,
                }
            ],
            "inventory": {}
        }

        # 1. Search all with alias 'gun'
        data = search_tracked_objects(q="gun", video_source="test_job_123", status_filter="all")
        assert data["success"] is True
        assert data["query"] == "gun"
        assert "Pistol" in data["canonical_classes"]
        assert data["matched_count"] == 1
        assert data["matching_tracks"][0]["track_id"] == 2

        # 2. Search with confirmed_only filter
        data_conf = search_tracked_objects(q="gun", video_source="test_job_123", status_filter="confirmed_only")
        assert data_conf["success"] is True
        assert data_conf["matched_count"] == 0  # Pistol was uncertain candidate

        # 3. Search with uncertain_only filter
        data_unc = search_tracked_objects(q="gun", video_source="test_job_123", status_filter="uncertain_only")
        assert data_unc["success"] is True
        assert data_unc["matched_count"] == 1

        # 4. Search person
        data_person = search_tracked_objects(q="person", video_source="test_job_123", status_filter="all")
        assert data_person["success"] is True
        assert data_person["matched_count"] == 1
        assert data_person["matching_tracks"][0]["detection_status"] == "confirmed_track"

    def test_export_endpoint_with_query_and_intervals(self):
        # JSON export
        resp_json = export_tracking_data(video_source="test_job_123", q="person", format="json")
        body_str = resp_json.body.decode("utf-8")
        data = json.loads(body_str)
        assert "tracks" in data
        assert len(data["tracks"]) == 1
        assert data["tracks"][0]["class_name"] == "person"
        assert "appearance_intervals" in data["tracks"][0]

        # CSV export
        resp_csv = export_tracking_data(video_source="test_job_123", q="person", format="csv")
        csv_text = resp_csv.body.decode("utf-8")
        assert "Track ID,Class Name,Category,Model Source" in csv_text
        assert "person" in csv_text
        assert "confirmed_track" in csv_text
        assert "1.0s – 2.5s" in csv_text
