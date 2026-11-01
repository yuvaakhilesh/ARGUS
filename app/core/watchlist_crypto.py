"""
ARGUS Phase 2 — Encryption Service.

Provides Fernet symmetric encryption/decryption for watchlist biometric data.

Key management:
  1. On first run, generate a key and store it in data/watchlist/.keyring
     (filesystem-level protection). In production, replace with a HSM or
     cloud KMS secret.
  2. The key is NEVER hardcoded and is NOT logged.
  3. If ARGUS_WATCHLIST_FERNET_KEY env var is set, it takes precedence
     (for container/cloud deployments).

A Fernet-encrypted blob is opaque; it contains ciphertext + HMAC-SHA256 + timestamp.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("argus.watchlist_crypto")

_KEY_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "watchlist" / ".keyring"

# Module-level singleton key object
_fernet = None
_fernet_lock = None


def _import_fernet():
    """Lazy import to avoid hard-crash if cryptography not yet installed."""
    try:
        from cryptography.fernet import Fernet, InvalidToken
        return Fernet, InvalidToken
    except ImportError:
        raise RuntimeError(
            "The 'cryptography' package is required for watchlist biometric encryption. "
            "Run: pip install cryptography"
        )


def _load_or_generate_key() -> bytes:
    """
    Load key from env var, then from keyring file.
    Generate and persist a new key if neither exists.
    """
    # Priority 1: environment variable (container / CI / KMS injection)
    env_key = os.environ.get("ARGUS_WATCHLIST_FERNET_KEY", "").strip()
    if env_key:
        logger.info("Watchlist encryption key loaded from environment variable.")
        return env_key.encode()

    # Priority 2: keyring file
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()

    # Generate new key
    Fernet, _ = _import_fernet()
    key = Fernet.generate_key()
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key)
    try:
        os.chmod(str(_KEY_FILE), 0o600)  # owner read-only
    except Exception:
        pass
    logger.warning(
        "New watchlist encryption key generated and saved to %s. "
        "Back this file up securely — loss of the key means loss of all watchlist biometric data.",
        _KEY_FILE,
    )
    return key


def get_fernet():
    """Return (or create) the module-level Fernet instance."""
    global _fernet, _fernet_lock
    import threading
    if _fernet_lock is None:
        _fernet_lock = threading.Lock()
    with _fernet_lock:
        if _fernet is None:
            Fernet, _ = _import_fernet()
            key = _load_or_generate_key()
            _fernet = Fernet(key)
    return _fernet


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt arbitrary bytes with Fernet. Returns ciphertext bytes."""
    return get_fernet().encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    """
    Decrypt Fernet token. Raises ValueError on decryption failure.
    Never logs plaintext.
    """
    _, InvalidToken = _import_fernet()
    try:
        return get_fernet().decrypt(token)
    except InvalidToken as exc:
        raise ValueError("Decryption failed: invalid or tampered token.") from exc


def encrypt_array(arr) -> bytes:
    """Encrypt a numpy float32 array to encrypted bytes."""
    import numpy as np
    buf = arr.astype(np.float32).tobytes()
    return encrypt_bytes(buf)


def decrypt_array(token: bytes, length: int = 512):
    """Decrypt bytes back to a numpy float32 array."""
    import numpy as np
    raw = decrypt_bytes(token)
    return np.frombuffer(raw, dtype=np.float32).copy()


def encrypt_image(img_bytes: bytes) -> bytes:
    """Encrypt raw image bytes (JPEG/PNG) for reference storage."""
    return encrypt_bytes(img_bytes)


def decrypt_image(token: bytes) -> bytes:
    """Decrypt encrypted reference image bytes."""
    return decrypt_bytes(token)

