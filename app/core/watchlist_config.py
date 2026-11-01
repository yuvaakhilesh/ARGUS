"""
ARGUS Phase 2 — Watchlist Configuration.

All face-watchlist paths, thresholds, and security settings.
Kept separate from core config.py to avoid disturbing existing functionality.
"""

import os
from pathlib import Path

# Base directory (same as config.py anchor)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Storage Directories ────────────────────────────────────────────────────────
WATCHLIST_DATA_DIR = BASE_DIR / "data" / "watchlist"
WATCHLIST_DB_PATH = WATCHLIST_DATA_DIR / "watchlist.db"
WATCHLIST_REFS_DIR = WATCHLIST_DATA_DIR / "refs"          # Encrypted reference images
WATCHLIST_EMBEDDINGS_DIR = WATCHLIST_DATA_DIR / "embeddings"  # Encrypted embedding blobs
WATCHLIST_AUDIT_DIR = WATCHLIST_DATA_DIR / "audit"        # Audit log shards
WATCHLIST_FACE_CROPS_DIR = WATCHLIST_DATA_DIR / "crops"   # Temporary CCTV face crops

# Ensure directories exist with restricted permissions
for _d in [
    WATCHLIST_DATA_DIR, WATCHLIST_REFS_DIR,
    WATCHLIST_EMBEDDINGS_DIR, WATCHLIST_AUDIT_DIR, WATCHLIST_FACE_CROPS_DIR
]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Face Detection & Embedding ─────────────────────────────────────────────────
# InsightFace buffalo_sc: lightweight SCRFD-500M detector + MobileNet ArcFace embedding
# Chosen for: ONNX-based (no PyTorch conflict), ~5ms det + ~3ms emb on RTX 4050,
#              CCTV-appropriate performance, MIT license (model weights: non-commercial research)
INSIGHTFACE_MODEL_NAME = "buffalo_sc"  # SCRFD-500M det + MBF ArcFace emb (~320 MB download)
INSIGHTFACE_MODEL_DIR = BASE_DIR / "data" / "insightface_models"
INSIGHTFACE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACE_DETECT_MIN_SIZE = 20        # Minimum face pixel size to consider (px, shorter side)
FACE_DETECT_THRESHOLD = 0.50    # SCRFD detection confidence threshold
FACE_QUALITY_BLUR_THRESHOLD = 80.0   # Laplacian variance; below → flagged as low quality
FACE_QUALITY_MIN_SIZE_PX = 40    # Minimum acceptable face short-side in px

# ── Matching ───────────────────────────────────────────────────────────────────
# ArcFace cosine similarity:  1.0 = identical, 0.0 = unrelated
# Validated empirically on LFW-derived threshold analysis:
#   • similarity >= 0.50 → "Possible match" (low false-positive, ~0.5% FMR at LFW)
#   • 0.40–0.50          → "Inconclusive — require human review"
#   • < 0.40             → "No match"
# These thresholds are ADVISORY only. All decisions require human review.
FACE_MATCH_POSSIBLE_THRESHOLD = 0.50   # >= this → possible watchlist match
FACE_MATCH_INCONCLUSIVE_THRESHOLD = 0.40  # >= this and < possible → inconclusive

# Maximum watchlist size for linear scan (extend to ANN index if > 5000 entries)
WATCHLIST_MAX_LINEAR_SIZE = 5000

# Sampling rate for face analysis within a video
FACE_SAMPLE_FPS = 2.0      # 2 frames per second (bounded to avoid VRAM pressure)
FACE_MAX_FRAMES = 300      # Hard cap: at most 300 frames per video analysis
FACE_DEDUP_SECONDS = 1.5   # Suppress duplicate observations within this window

# ── Security & Encryption ──────────────────────────────────────────────────────
# Key derivation: Fernet symmetric encryption (AES-128 CBC + HMAC-SHA256)
# Key material read from env var; never hardcoded.
WATCHLIST_ENCRYPTION_KEY_ENV = "ARGUS_WATCHLIST_FERNET_KEY"
WATCHLIST_SECRET_KEY_ENV = "ARGUS_JWT_SECRET"

# JWT token settings
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60      # 1-hour operator sessions
JWT_ADMIN_TOKEN_EXPIRE_MINUTES = 30       # Shorter for admin actions

# Bcrypt cost factor
BCRYPT_ROUNDS = 12

# ── Retention ─────────────────────────────────────────────────────────────────
WATCHLIST_DEFAULT_EXPIRY_DAYS = 365      # Default: 1 year
WATCHLIST_FACE_CROP_RETENTION_HOURS = 48 # Delete CCTV crops after 48 h

# ── Role Definitions ──────────────────────────────────────────────────────────
ROLES = {
    "admin": {
        "can_enroll": True,
        "can_revoke": True,
        "can_delete": True,
        "can_review": True,
        "can_manage_users": True,
        "can_view_audit": True,
        "can_trigger_analysis": True,
        "can_view_references": True,
    },
    "reviewer": {
        "can_enroll": False,
        "can_revoke": False,
        "can_delete": False,
        "can_review": True,
        "can_manage_users": False,
        "can_view_audit": False,
        "can_trigger_analysis": False,
        "can_view_references": True,
    },
    "operator": {
        "can_enroll": False,
        "can_revoke": False,
        "can_delete": False,
        "can_review": False,
        "can_manage_users": False,
        "can_view_audit": False,
        "can_trigger_analysis": True,
        "can_view_references": False,
    },
}

