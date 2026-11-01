"""
ARGUS Phase 2 — Watchlist Matching Engine.

Compares a probe face embedding against active watchlist entries.
Uses cosine similarity (standard ArcFace metric).

Threshold policy (advisory — NOT validated on ARGUS deployment footage):
  ≥ 0.50  → possible_match   (requires human review)
  ≥ 0.40  → inconclusive     (requires human review)
  < 0.40  → no_match

These thresholds are derived from published ArcFace LFW analysis.
They have NOT been independently validated on this deployment's camera
conditions, illumination profiles or subject population.
All outputs are advisory only.

Security:
  • Embeddings are decrypted in-process, used for comparison, then discarded.
  • Embeddings are never written to logs, JSON responses, or static files.
  • A similarity score is not a probability of identity.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import app.core.watchlist_db as db
from app.core.watchlist_config import (
    FACE_MATCH_INCONCLUSIVE_THRESHOLD,
    FACE_MATCH_POSSIBLE_THRESHOLD,
)
from app.core.watchlist_crypto import decrypt_array

logger = logging.getLogger("argus.watchlist_matcher")


@dataclass
class MatchCandidate:
    entry_id: str
    display_name: str
    similarity: float
    result: str  # 'possible_match' | 'inconclusive' | 'no_match'


@dataclass
class MatchResult:
    outcome: str          # 'possible_match' | 'no_match' | 'inconclusive' |
                          # 'insufficient_evidence' | 'processing_failure'
    top_match: Optional[MatchCandidate] = None
    all_candidates: List[MatchCandidate] = field(default_factory=list)
    advisory: str = (
        "AI-generated possible match. Identity has not been independently established. "
        "Human verification is required."
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Unit-normalized cosine similarity. ArcFace embeddings are L2-normalized."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_active_embeddings() -> List[Dict]:
    """
    Load and decrypt all active watchlist embeddings.
    Returns list of {entry_id, display_name, embedding(np.ndarray)}.
    Decryption errors are logged and the entry is skipped.
    """
    entries = db.list_active_watchlist(include_blobs=True)
    results = []
    for entry in entries:
        blob = entry.get("embedding_blob")
        if not blob:
            continue
        try:
            arr = decrypt_array(blob)
            results.append({
                "entry_id": entry["entry_id"],
                "display_name": entry["display_name"],
                "embedding": arr,
            })
        except Exception as exc:
            logger.warning(
                "Failed to decrypt embedding for entry %s: %s",
                entry["entry_id"], exc,
            )
    return results


def match_embedding(
    probe: np.ndarray,
    watchlist_entries: Optional[List[Dict]] = None,
) -> MatchResult:
    """
    Compare probe embedding against active watchlist.

    Args:
        probe: float32 (512,) ArcFace embedding (normalized or un-normalized).
        watchlist_entries: pre-loaded entries (optimization for batch analysis).
                           If None, loads fresh from DB.

    Returns:
        MatchResult with outcome and ordered candidate list.
    """
    if probe is None or len(probe) == 0:
        return MatchResult(outcome="insufficient_evidence")

    try:
        if watchlist_entries is None:
            watchlist_entries = _load_active_embeddings()

        if not watchlist_entries:
            return MatchResult(outcome="no_match")

        candidates: List[MatchCandidate] = []
        for entry in watchlist_entries:
            try:
                sim = _cosine_similarity(probe, entry["embedding"])
                if sim >= FACE_MATCH_POSSIBLE_THRESHOLD:
                    result = "possible_match"
                elif sim >= FACE_MATCH_INCONCLUSIVE_THRESHOLD:
                    result = "inconclusive"
                else:
                    result = "no_match"
                candidates.append(MatchCandidate(
                    entry_id=entry["entry_id"],
                    display_name=entry["display_name"],
                    similarity=round(sim, 4),
                    result=result,
                ))
            except Exception as exc:
                logger.debug("Error comparing probe to %s: %s", entry["entry_id"], exc)

        # Sort by descending similarity
        candidates.sort(key=lambda c: c.similarity, reverse=True)

        if not candidates:
            return MatchResult(outcome="no_match")

        top = candidates[0]

        if top.result == "possible_match":
            return MatchResult(outcome="possible_match", top_match=top, all_candidates=candidates)
        elif top.result == "inconclusive":
            return MatchResult(outcome="inconclusive", top_match=top, all_candidates=candidates)
        else:
            return MatchResult(outcome="no_match", all_candidates=candidates)

    except Exception as exc:
        logger.error("Matching engine error: %s", exc, exc_info=True)
        return MatchResult(outcome="processing_failure")


def batch_match_video(
    video_source: str,
    job_id: Optional[str],
    face_observations: List[Dict],
) -> List[str]:
    """
    Match multiple face observations from a video against the watchlist.
    Persists each observation in the DB and returns list of obs_ids.

    Args:
        face_observations: list of {
            source_timestamp, embedding(np.ndarray), quality, quality_score, face_crop_path
        }

    Returns: list of obs_ids
    """
    # Load watchlist once per video
    watchlist_entries = _load_active_embeddings()
    obs_ids = []

    for obs in face_observations:
        embedding = obs.get("embedding")
        quality = obs.get("quality", "unknown")
        quality_score = obs.get("quality_score")
        source_timestamp = obs.get("source_timestamp", 0.0)
        face_crop_path = obs.get("face_crop_path")

        if quality in ("tiny", "blurred") or embedding is None:
            match = MatchResult(outcome="insufficient_evidence")
        else:
            match = match_embedding(embedding, watchlist_entries)

        best_entry = match.top_match.entry_id if match.top_match else None
        best_sim = match.top_match.similarity if match.top_match else None
        all_cands = [
            {"entry_id": c.entry_id, "similarity": c.similarity, "result": c.result}
            for c in match.all_candidates
        ]

        oid = db.save_observation(
            video_source=video_source,
            source_timestamp=source_timestamp,
            job_id=job_id,
            face_crop_path=face_crop_path,
            image_quality=quality,
            quality_score=quality_score,
            best_match_entry=best_entry,
            best_similarity=best_sim,
            match_result=match.outcome,
            all_candidates=all_cands,
        )
        obs_ids.append(oid)

    return obs_ids

