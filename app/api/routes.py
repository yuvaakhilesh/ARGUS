import os
import shutil
import time
import uuid
import logging
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.core.config import (
    UPLOAD_DIR,
    SAMPLES_DIR,
    CLIPS_DIR,
    MODEL_NAME,
    CLASS_MAPPING,
    SEVERITY_MAPPING,
    CATEGORY_COLORS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_STRIDE_SECONDS,
    ALLOW_HEURISTIC_FALLBACK,
)
from app.core.model import VideoAnomalyDetector
from app.core.video_processor import VideoProcessor

logger = logging.getLogger("surveillance.api")

router = APIRouter()
detector = VideoAnomalyDetector.get_instance()
processor = VideoProcessor(detector)

# In-memory jobs repository
jobs: Dict[str, Dict[str, Any]] = {}

# Bounded inference concurrency: allows 1 heavy video analysis pipeline at a time on CPU
ANALYSIS_SEMAPHORE = threading.Semaphore(1)


def cleanup_temporary_files(max_age_hours: float = 24.0) -> int:
    """Removes orphaned temporary upload files older than max_age_hours."""
    cleaned_count = 0
    now = time.time()
    max_age_sec = max_age_hours * 3600.0
    active_paths = {j.get("video_path") for j in jobs.values() if j.get("status") == "processing"}
    try:
        for p in UPLOAD_DIR.glob("upload_*.*"):
            if str(p) in active_paths:
                continue
            try:
                if (now - p.stat().st_mtime) > max_age_sec:
                    p.unlink(missing_ok=True)
                    cleaned_count += 1
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")
    return cleaned_count


def run_analysis_worker(job_id: str, video_path: str, video_id: str):
    """Background worker for executing sliding-window VideoMAE analysis with progress tracking."""
    def progress_callback(stage: str, percentage: int):
        if job_id in jobs:
            jobs[job_id]["stage"] = stage
            jobs[job_id]["progress"] = percentage

    def cancel_check() -> bool:
        return jobs.get(job_id, {}).get("cancelled", False)

    try:
        if cancel_check():
            jobs[job_id]["status"] = "cancelled"
            Path(video_path).unlink(missing_ok=True)
            return

        progress_callback("Queued (waiting for available compute)", 18)
        with ANALYSIS_SEMAPHORE:
            if cancel_check():
                jobs[job_id]["status"] = "cancelled"
                Path(video_path).unlink(missing_ok=True)
                return

            progress_callback("Validating video quality", 20)

            results = processor.analyze_video(
                video_path=video_path,
                window_duration=DEFAULT_WINDOW_SECONDS,
                stride_duration=DEFAULT_STRIDE_SECONDS,
                video_id=video_id,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            if cancel_check() or results is None:
                jobs[job_id]["status"] = "cancelled"
                Path(video_path).unlink(missing_ok=True)
                return

            jobs[job_id]["status"] = "done"
            jobs[job_id]["stage"] = "Completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["results"] = results

    except Exception as e:
        logger.error(f"Analysis worker error for job {job_id}: {e}", exc_info=True)
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        # Safely remove uploaded video file on processing failure
        Path(video_path).unlink(missing_ok=True)


# -------------------------------------------------------------
# System Status & Diagnostic Endpoints
# -------------------------------------------------------------

@router.get("/status")
@router.get("/api/status")
def get_system_status():
    """Returns AI model status, active compute device, and UCF-Crime class configuration."""
    return {
        "status": "ready" if (detector.is_loaded or ALLOW_HEURISTIC_FALLBACK) else "error",
        "model_name": MODEL_NAME,
        "device": str(detector.device),
        "device_name": detector.device_name,
        "engine_mode": detector.engine_mode,
        "is_loaded": bool(detector.is_loaded),
        "pytorch_available": bool(detector.is_loaded),
        "loading_error": detector.loading_error,
        "allow_heuristic_fallback": ALLOW_HEURISTIC_FALLBACK,
        "classes_count": len(CLASS_MAPPING),
        "classes": list(CLASS_MAPPING.keys()),
        "severity_mapping": SEVERITY_MAPPING,
        "category_colors": CATEGORY_COLORS,
    }


@router.get("/samples")
@router.get("/api/samples")
def get_sample_videos():
    """Returns built-in surveillance clips for testing."""
    sample_items = [
        {
            "id": "sample_normal_patrol.mp4",
            "title": "Routine Security Patrol (Normal)",
            "camera": "CAM-01 [Main Corridor]",
            "expected_behavior": "Baseline normal pedestrian walking, zero security flags.",
            "url": "/static/samples/sample_normal_patrol.mp4",
            "tag": "NORMAL",
            "tag_color": "#10b981",
        },
        {
            "id": "sample_altercation_event.mp4",
            "title": "Parking Bay Altercation (High Alert)",
            "camera": "CAM-04 [Parking Bay 2]",
            "expected_behavior": "Sudden kinetic aggression and physical fighting detected at ~4s.",
            "url": "/static/samples/sample_altercation_event.mp4",
            "tag": "FIGHTING",
            "tag_color": "#e11d48",
        },
        {
            "id": "sample_night_intrusion.mp4",
            "title": "Infrared Perimeter Breach (Breach)",
            "camera": "CAM-09 [Rear Perimeter IR]",
            "expected_behavior": "Crouched stealthy intruder entering perimeter after dark.",
            "url": "/static/samples/sample_night_intrusion.mp4",
            "tag": "BURGLARY",
            "tag_color": "#a855f7",
        },
    ]
    return {"samples": sample_items}


# -------------------------------------------------------------
# Upload & Asynchronous Job Execution Endpoints
# -------------------------------------------------------------

@router.post("/upload")
@router.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives surveillance video (.mp4, .avi, .mov, .mkv, .webm), validates format and size,
    initiates background VideoMAE analysis, and returns a job_id.
    """
    valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted formats: {', '.join(valid_extensions)}"
        )

    if not detector.is_loaded and not ALLOW_HEURISTIC_FALLBACK:
        detector.load_model()
        if not detector.is_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"VideoMAE model is not loaded. Fallback heuristic is disabled. Error: {detector.loading_error}"
            )

    file_id = f"upload_{uuid.uuid4().hex[:8]}"
    save_filename = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / save_filename

    # Configurable maximum file size (default 25 GB)
    max_size_bytes = int(os.environ.get("ARGUS_MAX_UPLOAD_SIZE_BYTES", os.environ.get("AEGIS_MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024 * 1024)))
    chunk_size = 4 * 1024 * 1024  # 4 MB streaming chunks
    file_size = 0

    try:
        with open(save_path, "wb") as buffer:
            while True:
                # Bounded disk space check: require at least 100 MB free space
                free_disk = shutil.disk_usage(UPLOAD_DIR).free
                if free_disk < 100 * 1024 * 1024:
                    raise HTTPException(
                        status_code=507,
                        detail="Insufficient disk storage space on server to complete video upload."
                    )
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > max_size_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File exceeds maximum allowed size of {max_size_bytes / (1024*1024*1024):.1f} GB (size: {file_size / (1024*1024):.1f} MB)."
                    )
                buffer.write(chunk)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {str(e)}")

    if file_size == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        metadata = processor.get_video_metadata(str(save_path))
    except Exception as e:
        metadata = {"duration": 0.0, "fps": 25.0, "width": 0, "height": 0, "total_frames": 0}

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    jobs[job_id] = {
        "job_id": job_id,
        "video_id": file_id,
        "filename": save_filename,
        "original_name": file.filename,
        "url": f"/static/uploads/{save_filename}",
        "video_path": str(save_path),
        "file_size": file_size,
        "status": "processing",
        "stage": "Uploading",
        "progress": 15,
        "metadata": metadata,
        "results": None,
        "error": None,
        "cancelled": False,
        "created_at": time.time(),
    }

    # Start background analysis thread
    worker = threading.Thread(
        target=run_analysis_worker,
        args=(job_id, str(save_path), file_id),
        daemon=True,
    )
    worker.start()

    return {
        "success": True,
        "job_id": job_id,
        "video_id": file_id,
        "filename": save_filename,
        "original_name": file.filename,
        "url": f"/static/uploads/{save_filename}",
        "file_size": file_size,
        "metadata": metadata,
    }


@router.get("/status/{job_id}")
@router.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Returns the current progress, stage, and results of a video analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Analysis job '{job_id}' not found.")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],  # 'processing' | 'done' | 'cancelled' | 'error'
        "stage": job["stage"],
        "progress": job["progress"],
        "results": job["results"],
        "error": job["error"],
        "filename": job["filename"],
        "original_name": job.get("original_name"),
        "url": job["url"],
        "metadata": job.get("metadata"),
    }


@router.post("/cancel/{job_id}")
@router.post("/api/cancel/{job_id}")
def cancel_job(job_id: str):
    """Cancels an in-progress video analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    jobs[job_id]["cancelled"] = True
    jobs[job_id]["status"] = "cancelled"
    video_path = jobs[job_id].get("video_path")
    if video_path:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass
    return {"success": True, "job_id": job_id, "status": "cancelled"}


# -------------------------------------------------------------
# Clip Extraction Endpoints
# -------------------------------------------------------------

class CutClipRequest(BaseModel):
    job_id: Optional[str] = None
    video_source: Optional[str] = None
    start: float
    end: float
    label: Optional[str] = "EVIDENCE_CLIP"


@router.post("/clip")
@router.post("/api/clip")
def cut_and_download_clip(request_data: CutClipRequest, req: Request):
    """
    Extracts a precision trimmed H.264 MP4 clip server-side using FFmpeg.
    Returns the MP4 file for download/streaming or JSON metadata.
    """
    target_path = None

    if request_data.job_id and request_data.job_id in jobs:
        target_path = Path(jobs[request_data.job_id]["video_path"])
    elif request_data.video_source:
        source = request_data.video_source
        if (UPLOAD_DIR / source).exists():
            target_path = UPLOAD_DIR / source
        elif (SAMPLES_DIR / source).exists():
            target_path = SAMPLES_DIR / source
        elif Path(source).exists():
            target_path = Path(source)

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Target video not found.")

    clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
    out_path = CLIPS_DIR / clip_filename

    # Guard against whole-video download requests: strictly cap clip duration to max 30 seconds
    start_sec = max(0.0, float(request_data.start))
    end_sec = max(start_sec + 0.5, float(request_data.end))
    if end_sec - start_sec > 30.0:
        logger.warning(f"Clip request duration {end_sec - start_sec:.2f}s exceeds 30.0s max; capping.")
        end_sec = round(start_sec + 30.0, 2)

    success = processor.extract_video_clip(
        source_video_path=str(target_path),
        start_time=start_sec,
        end_time=end_sec,
        output_clip_path=out_path,
        overlay_text=f"{request_data.label} ({start_sec}s - {end_sec}s)",
    )

    if not success or not out_path.exists():
        raise HTTPException(status_code=500, detail="Failed to cut clip with FFmpeg.")

    # If client prefers JSON (e.g. Accept header or query param)
    accept_header = req.headers.get("accept", "")
    if "application/json" in accept_header and not req.query_params.get("download"):
        return {
            "success": True,
            "clip_filename": clip_filename,
            "clip_url": f"/static/clips/{clip_filename}",
            "start": request_data.start,
            "end": request_data.end,
        }

    # Return the MP4 file directly as downloadable attachment
    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=clip_filename,
        headers={"Content-Disposition": f'attachment; filename="{clip_filename}"'},
    )


# -------------------------------------------------------------
# Generalized Tracking & Weapon Analysis Endpoints
# -------------------------------------------------------------

# In-memory cache of latest tracking analysis results per video source
tracking_cache: Dict[str, Dict[str, Any]] = {}


def resolve_video_path(source: str) -> Optional[Path]:
    """Resolves video source name to path across uploads, samples, or absolute locations."""
    if not source:
        return None
    for d in [UPLOAD_DIR, SAMPLES_DIR]:
        if (d / source).exists():
            return d / source
    if Path(source).exists():
        return Path(source)
    clean = Path(source).name
    for d in [UPLOAD_DIR, SAMPLES_DIR]:
        if (d / clean).exists():
            return d / clean
    return None


class TrackingAnalysisRequest(BaseModel):
    video_source: str
    start_time: Optional[float] = 0.0
    end_time: Optional[float] = 10.0
    sample_fps: Optional[float] = 10.0
    run_weapons: Optional[bool] = True
    target_classes: Optional[List[str]] = None


@router.post("/tracking/analyze")
@router.post("/api/tracking/analyze")
def analyze_tracking_endpoint(request: TrackingAnalysisRequest):
    """
    On-demand generalized multi-object tracking and weapon detection endpoint.
    Runs GeneralizedTracker on specified video interval across all 80 COCO + 3 Weapon classes.
    """
    from app.core.tracking_pipeline import GeneralizedTracker
    target_path = resolve_video_path(request.video_source)
    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Target video '{request.video_source}' not found.")

    tracker = GeneralizedTracker.get_instance()
    results = tracker.analyze_clip(
        video_path=str(target_path),
        start_time=request.start_time or 0.0,
        end_time=request.end_time or 10.0,
        sample_fps=request.sample_fps or 10.0,
        run_weapons=bool(request.run_weapons),
        target_classes=request.target_classes,
    )
    # Cache for subsequent search, inventory, and track inspection
    tracking_cache[target_path.name] = results
    return {"success": True, "source": target_path.name, "results": results}


@router.get("/tracking/classes")
@router.get("/api/tracking/classes")
def get_tracking_classes():
    """
    Returns full dynamic catalog of all 83 supported detection classes
    (80 COCO classes + 3 dedicated tactical weapon classes), grouped by category.
    """
    from app.core.object_catalog import ObjectCatalog, CLASS_CATEGORIES, CATEGORY_COLORS
    classes = ObjectCatalog.get_supported_classes()
    return {
        "success": True,
        "total_classes": len(classes),
        "categories": list(CLASS_CATEGORIES.keys()),
        "category_colors": CATEGORY_COLORS,
        "classes": classes,
    }


@router.get("/tracking/objects")
@router.get("/api/tracking/objects")
def get_tracked_objects(
    video_source: Optional[str] = None,
    category: Optional[str] = None,
    class_name: Optional[str] = None,
    include_suppressed: bool = False,
):
    """
    Returns detected tracks for the selected video, with optional category/class filtering.
    """
    cached = None
    if video_source:
        clean = Path(video_source).name
        cached = tracking_cache.get(clean) or tracking_cache.get(video_source)
    if not cached and tracking_cache:
        cached = next(iter(tracking_cache.values()))

    if not cached:
        return {
            "success": True,
            "video_source": video_source,
            "status": "no_tracking_data",
            "message": "No tracking analysis has been run yet for this video.",
            "tracks": [],
            "total_tracks": 0,
        }

    tracks = cached.get("track_summaries", [])
    if not include_suppressed:
        tracks = [t for t in tracks if not t.get("is_suppressed")]

    if category:
        cat_lower = category.strip().lower()
        tracks = [t for t in tracks if t.get("category", "").lower() == cat_lower]

    if class_name:
        cls_lower = class_name.strip().lower()
        tracks = [t for t in tracks if t.get("class_name", "").lower() == cls_lower]

    return {
        "success": True,
        "video_source": video_source,
        "status": "completed",
        "total_tracks": len(tracks),
        "tracks": tracks,
    }


@router.get("/tracking/resolve-query")
@router.get("/api/tracking/resolve-query")
def resolve_query_endpoint(q: str):
    """
    Real-time query resolver providing natural language alias resolution and model class mappings.
    """
    from app.core.object_catalog import ObjectCatalog
    return {"success": True, **ObjectCatalog.resolve_query(q)}


@router.get("/tracking/search")
@router.get("/api/tracking/search")
def search_tracked_objects(
    q: str,
    video_source: Optional[str] = None,
    status_filter: Optional[str] = "all",
):
    """
    Fast Mode A search across existing analyzed tracks and detections.
    Supports natural language alias resolution, appearance intervals,
    and explicitly distinguishes between 'matches found', 'supported but not detected',
    and 'unsupported class'.
    """
    from app.core.tracking_pipeline import GeneralizedTracker
    from app.core.object_catalog import ObjectCatalog

    cached = None
    if video_source:
        clean = Path(video_source).name
        cached = tracking_cache.get(clean) or tracking_cache.get(video_source)
    if not cached and tracking_cache:
        cached = next(iter(tracking_cache.values()))

    resolved = ObjectCatalog.resolve_query(q)

    if not cached:
        return {
            "success": True,
            "query": q,
            "status": "no_analysis_data",
            "is_supported_class": resolved["is_supported"],
            "canonical_classes": resolved["canonical_classes"],
            "matched_aliases": resolved["matched_aliases"],
            "explanation": resolved["explanation"],
            "matched_count": 0,
            "matching_tracks": [],
            "confirmed_tracks_count": 0,
            "uncertain_candidates_count": 0,
            "message": "No video tracking analysis has been completed yet. Run tracking analysis first.",
        }

    tracks = cached.get("track_summaries", [])
    visible_tracks = [t for t in tracks if not t.get("is_suppressed")]
    tracker = GeneralizedTracker.get_instance()
    search_result = tracker.search_objects(visible_tracks, query=q, status_filter=status_filter or "all")

    return {"success": True, "video_source": video_source, **search_result}


@router.get("/tracking/tracks/{track_id}")
@router.get("/api/tracking/tracks/{track_id}")
def get_track_detail(track_id: str, video_source: Optional[str] = None):
    """
    Returns full telemetry and trajectory history for a specific track ID (e.g. OBJ-1, WEP-1).
    """
    cached = None
    if video_source:
        clean = Path(video_source).name
        cached = tracking_cache.get(clean) or tracking_cache.get(video_source)
    if not cached and tracking_cache:
        cached = next(iter(tracking_cache.values()))

    if not cached:
        raise HTTPException(status_code=404, detail="No tracking analysis data found.")

    tracks = cached.get("track_summaries", [])
    matched = next((t for t in tracks if t.get("track_id_str") == track_id or str(t.get("track_id")) == str(track_id)), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found in active video analysis.")

    # Find proximity cues involving this track
    cues = cached.get("weapon_analysis", {}).get("cues", [])
    related_cues = [
        c for c in cues
        if c.get("person_track_id") == track_id or c.get("weapon_track_id") == track_id
    ]

    return {
        "success": True,
        "track": matched,
        "related_cues": related_cues,
    }


class TargetedScanRequest(BaseModel):
    video_source: str
    target_classes: List[str]
    start_time: Optional[float] = 0.0
    end_time: Optional[float] = None
    sample_fps: Optional[float] = 10.0
    conf_threshold: Optional[float] = 0.25


@router.post("/tracking/scan")
@router.post("/api/tracking/scan")
def targeted_object_scan(request: TargetedScanRequest):
    """
    Mode B targeted scan for specific classes across a video interval.
    Accurately reports sampling rate, frames examined, and coverage.
    """
    from app.core.tracking_pipeline import GeneralizedTracker
    target_path = resolve_video_path(request.video_source)
    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Target video '{request.video_source}' not found.")

    tracker = GeneralizedTracker.get_instance()
    results = tracker.targeted_scan(
        video_path=str(target_path),
        target_classes=request.target_classes,
        start_time=request.start_time or 0.0,
        end_time=request.end_time,
        sample_fps=request.sample_fps or 10.0,
        conf_threshold=request.conf_threshold or 0.25,
    )
    return {"success": True, "results": results}


@router.get("/tracking/inventory")
@router.get("/api/tracking/inventory")
def get_tracking_inventory(video_source: Optional[str] = None):
    """
    Returns object inventory statistics (detections & unique tracks per class)
    for the analyzed video.
    """
    from app.core.object_catalog import ObjectCatalog
    cached = None
    if video_source:
        clean = Path(video_source).name
        cached = tracking_cache.get(clean) or tracking_cache.get(video_source)
    if not cached and tracking_cache:
        cached = next(iter(tracking_cache.values()))

    if not cached:
        return {
            "success": True,
            "video_source": video_source,
            "inventory": {
                "total_objects_tracked": 0,
                "distinct_classes_detected": 0,
                "category_summary": {},
                "inventory": [],
            },
        }

    inventory = cached.get("inventory")
    if not inventory:
        tracks = [t for t in cached.get("track_summaries", []) if not t.get("is_suppressed")]
        inventory = ObjectCatalog.build_inventory(tracks)

    return {"success": True, "video_source": video_source, "inventory": inventory}


@router.get("/tracking/export")
@router.get("/api/tracking/export")
def export_tracking_data(video_source: Optional[str] = None, q: Optional[str] = None, format: str = "json"):
    """
    Exports tracking observations and inventory in JSON or CSV format,
    with optional query filtering and appearance intervals.
    """
    import csv
    import io
    from app.core.tracking_pipeline import GeneralizedTracker

    cached = None
    if video_source:
        clean = Path(video_source).name
        cached = tracking_cache.get(clean) or tracking_cache.get(video_source)
    if not cached and tracking_cache:
        cached = next(iter(tracking_cache.values()))

    if not cached:
        raise HTTPException(status_code=404, detail="No tracking data available to export.")

    tracks = [t for t in cached.get("track_summaries", []) if not t.get("is_suppressed")]
    if q:
        tracker = GeneralizedTracker.get_instance()
        search_res = tracker.search_objects(tracks, query=q)
        tracks = search_res.get("matching_tracks", [])

    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Track ID", "Class Name", "Category", "Model Source",
            "Detection Status", "First Seen (s)", "Last Seen (s)", "Duration (s)",
            "Detection Count", "Max Confidence", "Avg Confidence", "Appearance Intervals", "Candidate Reason"
        ])
        for t in tracks:
            intervals_str = "; ".join([iv.get("display_text", "") for iv in t.get("appearance_intervals", [])])
            writer.writerow([
                t.get("track_id_str"),
                t.get("class_name"),
                t.get("category"),
                t.get("model_source"),
                t.get("detection_status", "confirmed_track"),
                t.get("first_seen_s"),
                t.get("last_seen_s"),
                t.get("duration_s"),
                t.get("detection_count"),
                t.get("max_confidence"),
                t.get("avg_confidence"),
                intervals_str,
                t.get("candidate_reason", ""),
            ])
        csv_content = output.getvalue()
        filename = f"argus_tracking_search_{q}.csv" if q else "argus_tracking_export.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Default JSON export
    filename = f"argus_tracking_search_{q}.json" if q else "argus_tracking_export.json"
    export_payload = {
        "export_timestamp": time.time(),
        "query": q,
        "video_source": video_source or cached.get("source"),
        "total_tracks": len(tracks),
        "inventory": cached.get("inventory"),
        "tracks": tracks,
        "weapon_analysis": cached.get("weapon_analysis"),
    }
    return JSONResponse(
        content=export_payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# -------------------------------------------------------------
# Legacy / Backward-Compatible Endpoints
# -------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    video_source: str
    window_seconds: Optional[float] = DEFAULT_WINDOW_SECONDS
    stride_seconds: Optional[float] = DEFAULT_STRIDE_SECONDS


@router.post("/analyze")
@router.post("/api/analyze")
def analyze_video_legacy(request: AnalyzeRequest):
    """Synchronous video analysis endpoint for legacy API callers."""
    source = request.video_source
    target_path = None

    if (UPLOAD_DIR / source).exists():
        target_path = UPLOAD_DIR / source
    elif (SAMPLES_DIR / source).exists():
        target_path = SAMPLES_DIR / source
    elif Path(source).exists():
        target_path = Path(source)
    else:
        clean_name = Path(source).name
        if (UPLOAD_DIR / clean_name).exists():
            target_path = UPLOAD_DIR / clean_name
        elif (SAMPLES_DIR / clean_name).exists():
            target_path = SAMPLES_DIR / clean_name

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{source}' not found.")

    if not detector.is_loaded and not ALLOW_HEURISTIC_FALLBACK:
        detector.load_model()
        if not detector.is_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"VideoMAE model is not loaded. Fallback heuristic is disabled. Error: {detector.loading_error}"
            )

    video_id = target_path.stem
    window_sec = max(1.0, min(request.window_seconds or DEFAULT_WINDOW_SECONDS, 10.0))
    stride_sec = max(0.5, min(request.stride_seconds or DEFAULT_STRIDE_SECONDS, 5.0))

    try:
        results = processor.analyze_video(
            video_path=str(target_path),
            window_duration=window_sec,
            stride_duration=stride_sec,
            video_id=video_id,
        )
        return {"success": True, "source": target_path.name, "analysis": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class ClipRequest(BaseModel):
    video_source: str
    start_time: float
    end_time: float
    label: Optional[str] = "EVIDENCE_CLIP"


@router.post("/extract-clip")
@router.post("/api/extract-clip")
def extract_custom_clip_legacy(request: ClipRequest):
    """Extracts custom clip returning JSON with clip_url."""
    source = request.video_source
    target_path = None
    if (UPLOAD_DIR / source).exists():
        target_path = UPLOAD_DIR / source
    elif (SAMPLES_DIR / source).exists():
        target_path = SAMPLES_DIR / source
    elif Path(source).exists():
        target_path = Path(source)

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{source}' not found.")

    clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
    out_path = CLIPS_DIR / clip_filename

    success = processor.extract_video_clip(
        source_video_path=str(target_path),
        start_time=request.start_time,
        end_time=request.end_time,
        output_clip_path=out_path,
        overlay_text=f"{request.label} ({request.start_time}s - {request.end_time}s)",
    )

    if not success or not out_path.exists():
        raise HTTPException(status_code=500, detail="Failed to extract clip")

    return {
        "success": True,
        "clip_filename": clip_filename,
        "clip_url": f"/static/clips/{clip_filename}",
        "start_time": request.start_time,
        "end_time": request.end_time,
    }
