"""Single-admin-credential auth for the dashboard.

Milestone 6. No user-management spec exists anywhere in the trial —
this is a single-operator internal tool (one team, one dashboard), so
auth is one admin credential from env vars plus a signed, expiring
session cookie. Not multi-user accounts; see the top-level README's setup
section for the required env vars.
"""

import os
from functools import lru_cache

from fastapi import Cookie, HTTPException
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext

SESSION_COOKIE_NAME = "autoace_session"
SESSION_MAX_AGE_S = 7 * 24 * 3600  # 7 days

_pwd_context = CryptContext(schemes=["bcrypt"])


@lru_cache(maxsize=1)
def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTOACE_SECRET_KEY")
    if not secret:
        # Dev fallback: a random key means sessions don't survive a
        # process restart. Fine for local dev, not for production — set
        # AUTOACE_SECRET_KEY there (see README).
        secret = os.urandom(32).hex()
    return URLSafeTimedSerializer(secret)


def _admin_username() -> str:
    return os.environ.get("AUTOACE_ADMIN_USERNAME", "admin")


@lru_cache(maxsize=1)
def _admin_password_hash() -> str:
    configured_hash = os.environ.get("AUTOACE_ADMIN_PASSWORD_HASH")
    if configured_hash:
        return configured_hash
    # Dev fallback: hash a plaintext env var at process start rather than
    # requiring operators to pre-generate a bcrypt hash for local testing.
    # Production deployments should set AUTOACE_ADMIN_PASSWORD_HASH instead.
    plaintext = os.environ.get("AUTOACE_ADMIN_PASSWORD", "autoace-dev-only")
    return _pwd_context.hash(plaintext)


def verify_credentials(username: str, password: str) -> bool:
    if username != _admin_username():
        return False
    return _pwd_context.verify(password, _admin_password_hash())


def create_session_cookie(username: str) -> str:
    return _serializer().dumps({"user": username})


def verify_session_cookie(cookie_value: str) -> str | None:
    try:
        data = _serializer().loads(cookie_value, max_age=SESSION_MAX_AGE_S)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("user")


def require_session(autoace_session: str | None = Cookie(default=None)) -> str:
    """FastAPI dependency — raises 401 unless a valid session cookie is present."""
    user = verify_session_cookie(autoace_session) if autoace_session else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
