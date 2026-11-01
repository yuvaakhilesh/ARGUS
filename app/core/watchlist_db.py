"""
ARGUS Phase 2 — Watchlist Database (SQLite).

Schema, migrations, and CRUD operations for:
  • users         — operator/reviewer/admin accounts
  • watchlist     — authorized biometric watchlist entries
  • observations  — CCTV face-match candidates
  • reviews       — human review decisions
  • audit_log     — immutable access and mutation audit trail

Design rules:
  • Raw face embeddings are NEVER stored in plain text.
    All blobs are Fernet-encrypted before write and decrypted on read.
  • Reference image bytes are stored as encrypted blobs; never accessible
    through static file serving.
  • Embedding bytes, reference bytes, and audit rows are never logged
    to stdout or returned in API responses without explicit authorization.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.watchlist_config import WATCHLIST_DB_PATH

logger = logging.getLogger("argus.watchlist_db")

# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    username       TEXT UNIQUE NOT NULL,
    password_hash  TEXT NOT NULL,           -- bcrypt
    role           TEXT NOT NULL CHECK(role IN ('admin','reviewer','operator')),
    full_name      TEXT NOT NULL DEFAULT '',
    is_active      INTEGER NOT NULL DEFAULT 1,
    created_at     REAL NOT NULL,
    last_login     REAL
);

CREATE TABLE IF NOT EXISTS watchlist (
    entry_id          TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL,
    reference_blob    BLOB,                 -- Fernet-encrypted JPEG bytes
    embedding_blob    BLOB,                 -- Fernet-encrypted float32 array
    enrollment_date   REAL NOT NULL,
    expiry_date       REAL,
    authorization_status  TEXT NOT NULL DEFAULT 'pending'
                      CHECK(authorization_status IN ('pending','active','revoked','expired','deleted')),
    authorization_basis   TEXT,
    enrolled_by       TEXT NOT NULL REFERENCES users(user_id),
    authorized_by     TEXT REFERENCES users(user_id),
    record_status     TEXT NOT NULL DEFAULT 'pending'
                      CHECK(record_status IN ('pending','active','expired','revoked','deleted')),
    notes             TEXT,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    obs_id            TEXT PRIMARY KEY,
    video_source      TEXT NOT NULL,
    source_timestamp  REAL NOT NULL,
    job_id            TEXT,
    face_crop_path    TEXT,                 -- local path; deleted after retention
    image_quality     TEXT NOT NULL DEFAULT 'unknown'
                      CHECK(image_quality IN ('good','low','blurred','tiny','unknown')),
    quality_score     REAL,
    best_match_entry  TEXT REFERENCES watchlist(entry_id),
    best_similarity   REAL,
    match_result      TEXT NOT NULL DEFAULT 'no_match'
                      CHECK(match_result IN (
                          'possible_match','no_match','inconclusive',
                          'insufficient_evidence','processing_failure'
                      )),
    all_candidates    TEXT,                 -- JSON: [{entry_id, similarity}]
    reviewed          INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id         TEXT PRIMARY KEY,
    obs_id            TEXT NOT NULL REFERENCES observations(obs_id),
    reviewer_id       TEXT NOT NULL REFERENCES users(user_id),
    decision          TEXT NOT NULL
                      CHECK(decision IN ('confirmed','rejected','inconclusive','further_review')),
    notes             TEXT,
    ai_similarity     REAL,
    review_timestamp  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id      TEXT PRIMARY KEY,
    timestamp   REAL NOT NULL,
    actor_id    TEXT,
    actor_role  TEXT,
    action      TEXT NOT NULL,
    target_type TEXT,
    target_id   TEXT,
    details     TEXT,                       -- JSON; never contains embedding bytes
    ip_address  TEXT
);

CREATE INDEX IF NOT EXISTS idx_watchlist_status ON watchlist(record_status);
CREATE INDEX IF NOT EXISTS idx_observations_video ON observations(video_source);
CREATE INDEX IF NOT EXISTS idx_observations_reviewed ON observations(reviewed);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_id);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe connection manager
# ─────────────────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(WATCHLIST_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_db():
    """Context manager yielding a thread-safe SQLite connection."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database():
    """Run schema migrations. Idempotent — safe to call on every startup."""
    with get_db() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Watchlist database initialized: %s", WATCHLIST_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def _now() -> float:
    return time.time()


def _audit(
    conn: sqlite3.Connection,
    action: str,
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[Dict] = None,
    ip_address: Optional[str] = None,
):
    """Append an immutable audit record inside the current transaction."""
    conn.execute(
        """INSERT INTO audit_log
           (log_id, timestamp, actor_id, actor_role, action,
            target_type, target_id, details, ip_address)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            _new_id("log_"),
            _now(),
            actor_id,
            actor_role,
            action,
            target_type,
            target_id,
            json.dumps(details or {}, default=str),
            ip_address,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────────────────────────────────────

def create_user(
    username: str,
    password_hash: str,
    role: str,
    full_name: str = "",
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    ip: Optional[str] = None,
) -> Dict:
    uid = _new_id("usr_")
    now = _now()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO users (user_id, username, password_hash, role, full_name, created_at)
               VALUES (?,?,?,?,?,?)""",
            (uid, username, password_hash, role, full_name, now),
        )
        _audit(conn, "user_created", actor_id, actor_role, "user", uid,
               {"username": username, "role": role}, ip)
    return {"user_id": uid, "username": username, "role": role}


def get_user_by_username(username: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT user_id, username, role, full_name, is_active, created_at, last_login"
            " FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def update_last_login(user_id: str):
    with get_db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE user_id=?", (_now(), user_id))


def list_users(actor_id: str, actor_role: str) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT user_id, username, role, full_name, is_active, created_at, last_login"
            " FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist CRUD
# ─────────────────────────────────────────────────────────────────────────────

def enroll_watchlist_entry(
    display_name: str,
    reference_blob: bytes,       # Already Fernet-encrypted
    embedding_blob: bytes,       # Already Fernet-encrypted
    expiry_date: Optional[float],
    authorization_basis: Optional[str],
    enrolled_by: str,
    notes: Optional[str] = None,
    actor_role: Optional[str] = None,
    ip: Optional[str] = None,
) -> Dict:
    eid = _new_id("wl_")
    now = _now()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO watchlist
               (entry_id, display_name, reference_blob, embedding_blob,
                enrollment_date, expiry_date, authorization_status,
                authorization_basis, enrolled_by, record_status,
                notes, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid, display_name, reference_blob, embedding_blob,
                now, expiry_date, "pending",
                authorization_basis, enrolled_by, "pending",
                notes, now, now,
            ),
        )
        _audit(conn, "watchlist_enrolled", enrolled_by, actor_role, "watchlist", eid,
               {"display_name": display_name}, ip)
    return {"entry_id": eid, "display_name": display_name, "record_status": "pending"}


def activate_watchlist_entry(
    entry_id: str,
    authorized_by: str,
    actor_role: str,
    ip: Optional[str] = None,
) -> bool:
    now = _now()
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE watchlist SET authorization_status='active', record_status='active',
               authorized_by=?, updated_at=?
               WHERE entry_id=? AND record_status='pending'""",
            (authorized_by, now, entry_id),
        )
        if cur.rowcount == 0:
            return False
        _audit(conn, "watchlist_activated", authorized_by, actor_role, "watchlist", entry_id, {}, ip)
    return True


def revoke_watchlist_entry(
    entry_id: str,
    actor_id: str,
    actor_role: str,
    reason: Optional[str] = None,
    ip: Optional[str] = None,
) -> bool:
    now = _now()
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE watchlist SET authorization_status='revoked', record_status='revoked',
               updated_at=?, notes=COALESCE(?,notes)
               WHERE entry_id=? AND record_status NOT IN ('deleted','revoked')""",
            (now, reason, entry_id),
        )
        if cur.rowcount == 0:
            return False
        _audit(conn, "watchlist_revoked", actor_id, actor_role, "watchlist", entry_id,
               {"reason": reason}, ip)
    return True


def delete_watchlist_entry(
    entry_id: str,
    actor_id: str,
    actor_role: str,
    ip: Optional[str] = None,
) -> bool:
    """
    Soft-delete: clears biometric blobs immediately, marks record as deleted.
    The audit trail entry is retained per legal requirement.
    """
    now = _now()
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE watchlist
               SET record_status='deleted', authorization_status='deleted',
                   reference_blob=NULL, embedding_blob=NULL, updated_at=?
               WHERE entry_id=? AND record_status != 'deleted'""",
            (now, entry_id),
        )
        if cur.rowcount == 0:
            return False
        _audit(conn, "watchlist_deleted_biometrics_purged", actor_id, actor_role,
               "watchlist", entry_id, {}, ip)
    return True


def get_watchlist_entry(entry_id: str, include_blobs: bool = False) -> Optional[Dict]:
    with get_db() as conn:
        if include_blobs:
            row = conn.execute("SELECT * FROM watchlist WHERE entry_id=?", (entry_id,)).fetchone()
        else:
            row = conn.execute(
                """SELECT entry_id, display_name, enrollment_date, expiry_date,
                          authorization_status, authorization_basis,
                          enrolled_by, authorized_by, record_status, notes, created_at, updated_at
                   FROM watchlist WHERE entry_id=?""",
                (entry_id,),
            ).fetchone()
    if not row:
        return None
    d = dict(row)
    # Sanitize: never leak blobs into non-admin contexts
    if not include_blobs:
        d.pop("reference_blob", None)
        d.pop("embedding_blob", None)
    return d


def list_active_watchlist(include_blobs: bool = False) -> List[Dict]:
    """Return active, non-expired entries for matching."""
    now = _now()
    with get_db() as conn:
        if include_blobs:
            rows = conn.execute(
                """SELECT * FROM watchlist
                   WHERE record_status='active'
                   AND (expiry_date IS NULL OR expiry_date > ?)
                   ORDER BY created_at DESC""",
                (now,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT entry_id, display_name, enrollment_date, expiry_date,
                          authorization_status, record_status, notes, created_at, updated_at
                   FROM watchlist
                   WHERE record_status='active'
                   AND (expiry_date IS NULL OR expiry_date > ?)
                   ORDER BY created_at DESC""",
                (now,),
            ).fetchall()
    result = [dict(r) for r in rows]
    if not include_blobs:
        for r in result:
            r.pop("reference_blob", None)
            r.pop("embedding_blob", None)
    return result


def list_all_watchlist() -> List[Dict]:
    """Admin view: all entries regardless of status."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT entry_id, display_name, enrollment_date, expiry_date,
                      authorization_status, record_status, notes, created_at, updated_at,
                      enrolled_by, authorized_by
               FROM watchlist ORDER BY created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def expire_stale_entries() -> int:
    """Mark entries past their expiry_date as expired. Called on startup."""
    now = _now()
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE watchlist SET record_status='expired', authorization_status='expired',
                   reference_blob=NULL, embedding_blob=NULL, updated_at=?
               WHERE record_status='active'
               AND expiry_date IS NOT NULL AND expiry_date <= ?""",
            (now, now),
        )
        count = cur.rowcount
        if count:
            _audit(conn, "watchlist_auto_expired_biometrics_purged", None, "system",
                   "watchlist", None, {"count": count})
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Observation & Review
# ─────────────────────────────────────────────────────────────────────────────

def save_observation(
    video_source: str,
    source_timestamp: float,
    job_id: Optional[str],
    face_crop_path: Optional[str],
    image_quality: str,
    quality_score: Optional[float],
    best_match_entry: Optional[str],
    best_similarity: Optional[float],
    match_result: str,
    all_candidates: Optional[List[Dict]],
) -> str:
    oid = _new_id("obs_")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO observations
               (obs_id, video_source, source_timestamp, job_id, face_crop_path,
                image_quality, quality_score, best_match_entry, best_similarity,
                match_result, all_candidates, reviewed, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (
                oid, video_source, source_timestamp, job_id, face_crop_path,
                image_quality, quality_score, best_match_entry, best_similarity,
                match_result,
                json.dumps(all_candidates or []),
                _now(),
            ),
        )
    return oid


def list_observations(
    video_source: Optional[str] = None,
    reviewed: Optional[bool] = None,
    limit: int = 200,
) -> List[Dict]:
    conditions = []
    params: List[Any] = []
    if video_source:
        conditions.append("video_source=?")
        params.append(video_source)
    if reviewed is not None:
        conditions.append("reviewed=?")
        params.append(1 if reviewed else 0)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT obs_id, video_source, source_timestamp, job_id, face_crop_path,
                       image_quality, quality_score, best_match_entry, best_similarity,
                       match_result, all_candidates, reviewed, created_at
                FROM observations {where} ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["all_candidates"] = json.loads(d["all_candidates"] or "[]")
        except Exception:
            d["all_candidates"] = []
        result.append(d)
    return result


def save_review(
    obs_id: str,
    reviewer_id: str,
    decision: str,
    notes: Optional[str],
    ai_similarity: Optional[float],
    actor_role: str,
    ip: Optional[str] = None,
) -> str:
    rid = _new_id("rev_")
    now = _now()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO reviews
               (review_id, obs_id, reviewer_id, decision, notes, ai_similarity, review_timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            (rid, obs_id, reviewer_id, decision, notes, ai_similarity, now),
        )
        conn.execute("UPDATE observations SET reviewed=1 WHERE obs_id=?", (obs_id,))
        _audit(conn, "observation_reviewed", reviewer_id, actor_role, "observation", obs_id,
               {"decision": decision}, ip)
    return rid


def get_observation_reviews(obs_id: str) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE obs_id=? ORDER BY review_timestamp DESC",
            (obs_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Audit Log
# ─────────────────────────────────────────────────────────────────────────────

def get_audit_log(
    limit: int = 500,
    actor_id: Optional[str] = None,
    target_id: Optional[str] = None,
) -> List[Dict]:
    conditions = []
    params: List[Any] = []
    if actor_id:
        conditions.append("actor_id=?")
        params.append(actor_id)
    if target_id:
        conditions.append("target_id=?")
        params.append(target_id)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]

