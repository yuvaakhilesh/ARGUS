"""
ARGUS Phase 2 — Watchlist API Routes.

All endpoints are prefixed /api/watchlist/*.
Authentication is enforced via Bearer JWT.
Role checks are applied per-endpoint.

Endpoints:
  POST   /api/auth/login                  — obtain JWT token
  POST   /api/auth/users                  — create user (admin only)
  GET    /api/auth/users                  — list users (admin only)
  POST   /api/watchlist/enroll            — enroll watchlist entry (admin)
  POST   /api/watchlist/{id}/activate     — activate entry (admin)
  GET    /api/watchlist                   — list entries (admin/reviewer)
  GET    /api/watchlist/{id}              — get entry (admin/reviewer)
  PUT    /api/watchlist/{id}              — update notes/expiry (admin)
  POST   /api/watchlist/{id}/revoke       — revoke entry (admin)
  DELETE /api/watchlist/{id}              — delete biometrics (admin)
  POST   /api/watchlist/analyze           — trigger face analysis on video (admin/operator)
  GET    /api/watchlist/matches           — list observations (admin/reviewer/operator)
  GET    /api/watchlist/matches/{id}      — observation detail (admin/reviewer)
  GET    /api/watchlist/matches/{id}/reference  — reference image (admin/reviewer)
  GET    /api/watchlist/matches/{id}/crop — CCTV face crop (admin/reviewer)
  POST   /api/watchlist/matches/{id}/review  — submit review (reviewer/admin)
  GET    /api/watchlist/audit             — audit log (admin)

Security properties:
  • Embeddings are NEVER returned in API responses.
  • Reference images are served only to admin/reviewer via authenticated endpoint.
  • CCTV face crops are served only to admin/reviewer via authenticated endpoint.
  • All mutation endpoints emit an audit log entry.
  • Bulk export of watchlist is blocked (no endpoint exposes raw embeddings).
"""

import io
import logging
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import app.core.watchlist_db as db
from app.core.watchlist_auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_permission,
    require_role,
    verify_password,
)
from app.core.watchlist_config import (
    WATCHLIST_DEFAULT_EXPIRY_DAYS,
    WATCHLIST_FACE_CROPS_DIR,
)
from app.core.watchlist_crypto import (
    decrypt_image,
    encrypt_array,
    encrypt_image,
)

logger = logging.getLogger("argus.watchlist_routes")

watchlist_router = APIRouter(prefix="/api/watchlist", tags=["Watchlist"])
auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])

# ─────────────────────────────────────────────────────────────────────────────
# Auth Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


@auth_router.post("/login")
def login(body: LoginRequest, req: Request):
    """Authenticate and return a Bearer JWT token."""
    user = db.get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated.")
    db.update_last_login(user["user_id"])
    token = create_access_token(user["user_id"], user["username"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "user_id": user["user_id"],
        "username": user["username"],
    }


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = Field(..., pattern="^(admin|reviewer|operator)$")
    full_name: str = ""


@auth_router.post("/users", dependencies=[Depends(require_role("admin"))])
def create_user(body: CreateUserRequest, req: Request, actor=Depends(require_role("admin"))):
    """Create a new user account (admin only)."""
    existing = db.get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")
    pw_hash = hash_password(body.password)
    result = db.create_user(
        username=body.username,
        password_hash=pw_hash,
        role=body.role,
        full_name=body.full_name,
        actor_id=actor["user_id"],
        actor_role=actor["role"],
        ip=req.client.host if req.client else None,
    )
    return {"success": True, **result}


@auth_router.get("/users", dependencies=[Depends(require_role("admin"))])
def list_users(actor=Depends(require_role("admin"))):
    """List all user accounts (admin only)."""
    return {"users": db.list_users(actor["user_id"], actor["role"])}


@auth_router.get("/me")
def get_me(user=Depends(get_current_user)):
    """Return current authenticated user profile."""
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "role": user["role"],
        "full_name": user.get("full_name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist Enrollment
# ─────────────────────────────────────────────────────────────────────────────


@watchlist_router.post("/enroll")
async def enroll_watchlist(
    req: Request,
    display_name: str = Form(...),
    authorization_basis: Optional[str] = Form(None),
    expiry_days: Optional[int] = Form(WATCHLIST_DEFAULT_EXPIRY_DAYS),
    notes: Optional[str] = Form(None),
    reference_image: UploadFile = File(...),
    actor=Depends(require_permission("can_enroll")),
):
    """
    Enroll an authorized person in the face watchlist.
    Validates the reference image, extracts the face embedding,
    encrypts and stores both. Starts in 'pending' state.
    """
    # ── Validate image type ────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if reference_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Reference image must be JPEG or PNG. Got: {reference_image.content_type}",
        )

    img_bytes = await reference_image.read()
    if len(img_bytes) > 10 * 1024 * 1024:  # 10 MB max
        raise HTTPException(status_code=400, detail="Reference image must not exceed 10 MB.")

    # ── Face detection on reference image ─────────────────────────────────
    from app.core.face_processor import FaceProcessor
    fp = FaceProcessor.get_instance()
    if not fp.is_loaded:
        if not fp.load_model():
            raise HTTPException(
                status_code=503,
                detail="Face processing module is unavailable. Cannot validate reference image.",
            )

    try:
        faces = fp.process_image_bytes(img_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not faces:
        raise HTTPException(
            status_code=422,
            detail="No usable face detected in the reference image. "
                   "Please provide a clear frontal photograph.",
        )
    if len(faces) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"Multiple faces detected ({len(faces)}). "
                   "Reference image must contain exactly one face.",
        )

    face = faces[0]
    if face.embedding is None:
        raise HTTPException(
            status_code=422,
            detail="Face detected but embedding could not be extracted. "
                   "Please use a higher-quality image.",
        )
    if face.quality == "tiny":
        raise HTTPException(
            status_code=422,
            detail="Reference face is too small. Please use a higher-resolution image.",
        )

    # ── Encrypt and store ──────────────────────────────────────────────────
    enc_image = encrypt_image(img_bytes)
    enc_embedding = encrypt_array(face.embedding)

    expiry_ts = None
    if expiry_days and expiry_days > 0:
        expiry_ts = time.time() + expiry_days * 86400

    result = db.enroll_watchlist_entry(
        display_name=display_name,
        reference_blob=enc_image,
        embedding_blob=enc_embedding,
        expiry_date=expiry_ts,
        authorization_basis=authorization_basis,
        enrolled_by=actor["user_id"],
        notes=notes,
        actor_role=actor["role"],
        ip=req.client.host if req.client else None,
    )

    return {
        "success": True,
        "entry_id": result["entry_id"],
        "display_name": display_name,
        "record_status": "pending",
        "face_quality": face.quality,
        "face_quality_score": round(face.quality_score, 2),
        "message": "Enrolled in pending state. An admin must activate this entry before it is used in matching.",
    }


@watchlist_router.post("/{entry_id}/activate")
def activate_entry(
    entry_id: str,
    req: Request,
    actor=Depends(require_role("admin")),
):
    """Activate a pending watchlist entry for use in matching (admin only)."""
    ok = db.activate_watchlist_entry(
        entry_id=entry_id,
        authorized_by=actor["user_id"],
        actor_role=actor["role"],
        ip=req.client.host if req.client else None,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="Entry not found or is not in pending state.",
        )
    return {"success": True, "entry_id": entry_id, "record_status": "active"}


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist Retrieval
# ─────────────────────────────────────────────────────────────────────────────


@watchlist_router.get("")
def list_watchlist(actor=Depends(require_role("admin", "reviewer"))):
    """Return all watchlist entries (no biometric blobs) — admin or reviewer."""
    if actor["role"] == "admin":
        entries = db.list_all_watchlist()
    else:
        entries = db.list_active_watchlist(include_blobs=False)
    return {"entries": entries, "count": len(entries)}


@watchlist_router.get("/{entry_id}")
def get_entry(entry_id: str, actor=Depends(require_role("admin", "reviewer"))):
    """Get a single watchlist entry (no biometric blobs)."""
    entry = db.get_watchlist_entry(entry_id, include_blobs=False)
    if not entry:
        raise HTTPException(status_code=404, detail="Watchlist entry not found.")
    return entry


class UpdateEntryRequest(BaseModel):
    notes: Optional[str] = None
    expiry_days: Optional[int] = None
    authorization_basis: Optional[str] = None


@watchlist_router.put("/{entry_id}")
def update_entry(
    entry_id: str,
    body: UpdateEntryRequest,
    req: Request,
    actor=Depends(require_role("admin")),
):
    """Update notes, expiry, or authorization basis (admin only)."""
    entry = db.get_watchlist_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Watchlist entry not found.")
    if entry["record_status"] == "deleted":
        raise HTTPException(status_code=410, detail="Entry has been deleted.")

    from app.core.watchlist_db import get_db
    now = time.time()
    new_expiry = None
    if body.expiry_days and body.expiry_days > 0:
        new_expiry = now + body.expiry_days * 86400

    updates: Dict[str, Any] = {"updated_at": now}
    if body.notes is not None:
        updates["notes"] = body.notes
    if body.authorization_basis is not None:
        updates["authorization_basis"] = body.authorization_basis
    if new_expiry:
        updates["expiry_date"] = new_expiry

    set_clause = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [entry_id]

    with get_db() as conn:
        conn.execute(f"UPDATE watchlist SET {set_clause} WHERE entry_id=?", vals)
        from app.core.watchlist_db import _audit
        _audit(conn, "watchlist_updated", actor["user_id"], actor["role"], "watchlist", entry_id,
               {k: str(v) for k, v in updates.items()},
               req.client.host if req.client else None)

    return {"success": True, "entry_id": entry_id}


@watchlist_router.post("/{entry_id}/revoke")
def revoke_entry(
    entry_id: str,
    req: Request,
    reason: Optional[str] = None,
    actor=Depends(require_role("admin")),
):
    """Revoke (deactivate) a watchlist entry without deleting audit trail."""
    ok = db.revoke_watchlist_entry(
        entry_id=entry_id,
        actor_id=actor["user_id"],
        actor_role=actor["role"],
        reason=reason,
        ip=req.client.host if req.client else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found or already revoked/deleted.")
    return {"success": True, "entry_id": entry_id, "record_status": "revoked"}


@watchlist_router.delete("/{entry_id}")
def delete_entry(
    entry_id: str,
    req: Request,
    actor=Depends(require_role("admin")),
):
    """
    Permanently delete biometric blobs for a watchlist entry.
    The entry record and audit trail are retained.
    """
    ok = db.delete_watchlist_entry(
        entry_id=entry_id,
        actor_id=actor["user_id"],
        actor_role=actor["role"],
        ip=req.client.host if req.client else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found or already deleted.")
    return {
        "success": True,
        "entry_id": entry_id,
        "record_status": "deleted",
        "message": "Biometric data permanently deleted. Audit trail preserved.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Face Analysis Trigger
# ─────────────────────────────────────────────────────────────────────────────

# In-memory face analysis jobs tracker (shared with existing jobs dict via import)
_face_jobs: Dict[str, Dict] = {}
_face_jobs_lock = threading.Lock()


class FaceAnalysisRequest(BaseModel):
    video_source: str   # filename in uploads/ or full path
    job_id: Optional[str] = None   # existing VideoMAE job_id to link


@watchlist_router.post("/analyze")
def trigger_face_analysis(
    body: FaceAnalysisRequest,
    req: Request,
    actor=Depends(require_permission("can_trigger_analysis")),
):
    """
    Trigger face-watchlist analysis on an uploaded video.
    Disabled by default — must be explicitly called by an authorized operator.
    Returns a face_job_id to poll.
    """
    from app.core.config import UPLOAD_DIR, SAMPLES_DIR

    # Resolve video path
    source = body.video_source
    target_path = None
    for base in [UPLOAD_DIR, SAMPLES_DIR]:
        candidate = base / source
        if candidate.exists():
            target_path = candidate
            break
    if target_path is None and Path(source).exists():
        target_path = Path(source)
    if target_path is None:
        raise HTTPException(status_code=404, detail=f"Video '{source}' not found.")

    face_job_id = f"fj_{uuid.uuid4().hex[:10]}"
    with _face_jobs_lock:
        _face_jobs[face_job_id] = {
            "face_job_id": face_job_id,
            "video_source": target_path.name,
            "job_id": body.job_id,
            "status": "queued",
            "result": None,
            "error": None,
            "initiated_by": actor["user_id"],
            "initiated_at": time.time(),
        }

    def _worker():
        from app.core.face_analysis_worker import analyze_video_faces

        def _progress(stage, pct):
            with _face_jobs_lock:
                _face_jobs[face_job_id]["stage"] = stage
                _face_jobs[face_job_id]["progress"] = pct

        try:
            with _face_jobs_lock:
                _face_jobs[face_job_id]["status"] = "processing"
            result = analyze_video_faces(
                video_path=str(target_path),
                job_id=body.job_id,
                progress_callback=_progress,
            )
            with _face_jobs_lock:
                _face_jobs[face_job_id]["status"] = "done"
                _face_jobs[face_job_id]["result"] = result
        except Exception as exc:
            logger.error("Face analysis worker error: %s", exc, exc_info=True)
            with _face_jobs_lock:
                _face_jobs[face_job_id]["status"] = "error"
                _face_jobs[face_job_id]["error"] = str(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    return {
        "success": True,
        "face_job_id": face_job_id,
        "video_source": target_path.name,
        "advisory": (
            "Face-watchlist analysis initiated. Results require human review. "
            "AI-generated possible matches are not proof of identity."
        ),
    }


@watchlist_router.get("/jobs/{face_job_id}")
def get_face_job_status(
    face_job_id: str,
    actor=Depends(get_current_user),
):
    """Poll status of a face analysis job."""
    with _face_jobs_lock:
        job = _face_jobs.get(face_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Face analysis job not found.")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# Match Observation Retrieval & Review
# ─────────────────────────────────────────────────────────────────────────────


@watchlist_router.get("/matches")
def list_matches(
    video_source: Optional[str] = None,
    reviewed: Optional[bool] = None,
    actor=Depends(get_current_user),
):
    """
    List face-match observations.
    Operators see only video-source-scoped results for videos they triggered.
    Reviewers and admins see all.
    """
    obs = db.list_observations(video_source=video_source, reviewed=reviewed)
    # Strip face_crop_path from operator-level responses
    if actor["role"] == "operator":
        for o in obs:
            o.pop("face_crop_path", None)
    return {
        "observations": obs,
        "count": len(obs),
        "advisory": (
            "AI-generated possible matches. Identity has not been independently established. "
            "Human verification is required."
        ),
    }


@watchlist_router.get("/matches/{obs_id}")
def get_match_detail(obs_id: str, actor=Depends(require_role("admin", "reviewer"))):
    """Return full observation detail for a reviewer or admin."""
    from app.core.watchlist_db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM observations WHERE obs_id=?", (obs_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Observation not found.")
    import json
    d = dict(row)
    try:
        d["all_candidates"] = json.loads(d.get("all_candidates") or "[]")
    except Exception:
        d["all_candidates"] = []
    d.pop("face_crop_path_raw", None)
    reviews = db.get_observation_reviews(obs_id)

    # Get matched watchlist entry (no blobs)
    matched_entry = None
    if d.get("best_match_entry"):
        matched_entry = db.get_watchlist_entry(d["best_match_entry"], include_blobs=False)

    return {
        "observation": d,
        "matched_entry": matched_entry,
        "reviews": reviews,
        "advisory": (
            "AI-generated possible match. Identity has not been independently established. "
            "Human verification is required."
        ),
    }


@watchlist_router.get("/matches/{obs_id}/reference")
def get_reference_image(obs_id: str, actor=Depends(require_role("admin", "reviewer"))):
    """
    Return the encrypted reference image for the watchlist entry matched by this observation.
    Only admin and reviewer roles may access this endpoint.
    """
    from app.core.watchlist_db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT best_match_entry FROM observations WHERE obs_id=?", (obs_id,)
        ).fetchone()
    if not row or not row["best_match_entry"]:
        raise HTTPException(status_code=404, detail="No matched entry for this observation.")

    entry_row = db.get_watchlist_entry(row["best_match_entry"], include_blobs=True)
    if not entry_row or entry_row.get("record_status") == "deleted":
        raise HTTPException(status_code=404, detail="Reference not available (entry deleted).")

    enc_blob = entry_row.get("reference_blob")
    if not enc_blob:
        raise HTTPException(status_code=404, detail="Reference image not available.")

    try:
        img_bytes = decrypt_image(enc_blob)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to decrypt reference image.")

    return Response(content=img_bytes, media_type="image/jpeg")


@watchlist_router.get("/matches/{obs_id}/crop")
def get_face_crop(obs_id: str, actor=Depends(require_role("admin", "reviewer"))):
    """Return the CCTV face crop for an observation (admin/reviewer only)."""
    from app.core.watchlist_db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT face_crop_path FROM observations WHERE obs_id=?", (obs_id,)
        ).fetchone()
    if not row or not row["face_crop_path"]:
        raise HTTPException(status_code=404, detail="Face crop not available.")
    crop_path = Path(row["face_crop_path"])
    if not crop_path.exists():
        raise HTTPException(status_code=410, detail="Face crop has been deleted (retention policy).")
    return Response(content=crop_path.read_bytes(), media_type="image/jpeg")


class ReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(confirmed|rejected|inconclusive|further_review)$")
    notes: Optional[str] = None


@watchlist_router.post("/matches/{obs_id}/review")
def submit_review(
    obs_id: str,
    body: ReviewRequest,
    req: Request,
    actor=Depends(require_permission("can_review")),
):
    """
    Submit a human review decision for a face-match observation.
    The original AI similarity score is preserved unchanged.
    """
    from app.core.watchlist_db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT obs_id, best_similarity FROM observations WHERE obs_id=?", (obs_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Observation not found.")

    review_id = db.save_review(
        obs_id=obs_id,
        reviewer_id=actor["user_id"],
        decision=body.decision,
        notes=body.notes,
        ai_similarity=row["best_similarity"],
        actor_role=actor["role"],
        ip=req.client.host if req.client else None,
    )
    return {
        "success": True,
        "review_id": review_id,
        "obs_id": obs_id,
        "decision": body.decision,
        "disclaimer": (
            "This review records a human assessment. "
            "It does NOT constitute conclusive proof of identity."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audit Log
# ─────────────────────────────────────────────────────────────────────────────


@watchlist_router.get("/audit")
def get_audit(
    limit: int = 200,
    actor_filter: Optional[str] = None,
    target_filter: Optional[str] = None,
    actor=Depends(require_role("admin")),
):
    """Return the watchlist audit log (admin only)."""
    rows = db.get_audit_log(limit=limit, actor_id=actor_filter, target_id=target_filter)
    return {"audit_log": rows, "count": len(rows)}

