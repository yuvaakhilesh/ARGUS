"""
ARGUS Phase 2 — Authentication & Authorization.

Provides:
  • bcrypt password hashing and verification.
  • JWT access token creation and validation (HS256).
  • FastAPI dependency for authenticated endpoints.
  • Role-permission checking.
  • Bootstrap: auto-creates the initial admin account from env vars on startup.

JWT secret is read from ARGUS_JWT_SECRET env var.
If absent, a random secret is generated per process.
A stable secret across restarts requires the env var to be set.
"""

import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.watchlist_config import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    ROLES,
    WATCHLIST_SECRET_KEY_ENV,
)
import app.core.watchlist_db as db

logger = logging.getLogger("argus.watchlist_auth")

# ── JWT secret ────────────────────────────────────────────────────────────────

_jwt_secret: Optional[str] = None


def _get_jwt_secret() -> str:
    global _jwt_secret
    if _jwt_secret is None:
        env_secret = os.environ.get(WATCHLIST_SECRET_KEY_ENV, "").strip()
        if env_secret:
            _jwt_secret = env_secret
        else:
            _jwt_secret = secrets.token_hex(32)
            logger.warning(
                "ARGUS_JWT_SECRET not set. Using a per-process random secret — "
                "all tokens will be invalidated on server restart. "
                "Set ARGUS_JWT_SECRET in production."
            )
    return _jwt_secret


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    try:
        import bcrypt  # type: ignore
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(plain.encode(), salt).decode()
    except ImportError:
        # Fallback: pbkdf2 via hashlib (no extra dependency, less strong)
        import hashlib
        logger.warning("bcrypt not available; using PBKDF2-SHA256 fallback.")
        import os as _os
        _salt = _os.urandom(16).hex()
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), _salt.encode(), 260000)
        return f"pbkdf2:{_salt}:{dk.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    try:
        import bcrypt  # type: ignore
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ImportError:
        import hashlib
        if not hashed.startswith("pbkdf2:"):
            return False
        _, _salt, _dk = hashed.split(":", 2)
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), _salt.encode(), 260000)
        return dk.hex() == _dk


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: str, username: str, role: str) -> str:
    try:
        from jose import jwt  # type: ignore
    except ImportError:
        raise RuntimeError(
            "python-jose is required for JWT auth. Run: pip install python-jose"
        )
    expire = time.time() + JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": int(expire),
        "iat": int(time.time()),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        from jose import jwt, JWTError  # type: ignore
    except ImportError:
        raise RuntimeError("python-jose required.")
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        if time.time() > payload.get("exp", 0):
            return None
        return payload
    except Exception:
        return None


# ── FastAPI bearer dependency ─────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


def _credentials_exception(detail: str = "Not authenticated"):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    if creds is None:
        raise _credentials_exception()
    payload = decode_access_token(creds.credentials)
    if payload is None:
        raise _credentials_exception("Token invalid or expired.")
    user = db.get_user_by_id(payload["sub"])
    if not user or not user.get("is_active"):
        raise _credentials_exception("User not found or deactivated.")
    return user


def require_role(*allowed_roles: str):
    """Returns a FastAPI dependency that enforces one of the allowed roles."""
    def _check(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' is not authorized for this action.",
            )
        return user
    return _check


def require_permission(permission: str):
    """Returns a FastAPI dependency that checks a named permission in ROLES."""
    def _check(user: dict = Depends(get_current_user)):
        role_perms = ROLES.get(user["role"], {})
        if not role_perms.get(permission, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required.",
            )
        return user
    return _check


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_admin():
    """
    Create initial admin account from environment variables if no admin exists.
    Env vars: ARGUS_ADMIN_USERNAME, ARGUS_ADMIN_PASSWORD
    If not set, defaults to admin / argus-admin-changeme with a warning.
    """
    admin_username = os.environ.get("ARGUS_ADMIN_USERNAME", "admin").strip()
    admin_password = os.environ.get("ARGUS_ADMIN_PASSWORD", "argus-admin-changeme").strip()

    if os.environ.get("ARGUS_ADMIN_PASSWORD", "") == "":
        logger.critical(
            "ARGUS_ADMIN_PASSWORD not set! Using default password 'argus-admin-changeme'. "
            "CHANGE THIS IMMEDIATELY in production."
        )

    existing = db.get_user_by_username(admin_username)
    if existing:
        return  # Admin already exists

    pw_hash = hash_password(admin_password)
    db.create_user(
        username=admin_username,
        password_hash=pw_hash,
        role="admin",
        full_name="System Administrator",
        actor_id=None,
        actor_role="system",
    )
    logger.info("Bootstrapped initial admin account: %s", admin_username)

