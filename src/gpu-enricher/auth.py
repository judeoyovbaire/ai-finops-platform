"""
API Authentication Module for AI FinOps Platform

Provides API key authentication and rate limiting for the GPU Enricher API.
Supports multiple authentication methods:
- API Key via header (X-API-Key)
- API Key via query parameter (api_key)
- Bearer token authentication
"""

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

from flask import jsonify, request

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API key configuration."""

    key_hash: str  # SHA-256 hash of the key
    name: str
    team: Optional[str] = None
    scopes: list[str] = field(default_factory=lambda: ["read"])
    rate_limit: int = 100  # Requests per minute
    enabled: bool = True


@dataclass
class RateLimitState:
    """Rate limiting state for an API key."""

    requests: list[float] = field(default_factory=list)
    window_seconds: int = 60


class AuthManager:
    """Manages API authentication and rate limiting."""

    HEADER_NAME = "X-API-Key"
    QUERY_PARAM = "api_key"
    BEARER_PREFIX = "Bearer "

    def __init__(self):
        self._api_keys: dict[str, APIKey] = {}
        self._rate_limits: dict[str, RateLimitState] = {}
        self._enabled = False
        self._require_auth_for_write = True

    def initialize(self) -> None:
        """Initialize auth from environment variables."""
        # Check if auth is enabled
        self._enabled = os.getenv("ENABLE_API_AUTH", "false").lower() == "true"
        self._require_auth_for_write = (
            os.getenv("REQUIRE_AUTH_FOR_WRITE", "true").lower() == "true"
        )

        if not self._enabled:
            logger.info("API authentication disabled")
            return

        # Load API keys from environment
        # Format: API_KEY_<name>=<key>
        # Format: API_KEY_<name>_SCOPES=read,write
        # Format: API_KEY_<name>_TEAM=team-name
        for key, value in os.environ.items():
            if key.startswith("API_KEY_") and not key.endswith(("_SCOPES", "_TEAM", "_RATE_LIMIT")):
                name = key[8:]  # Remove "API_KEY_" prefix
                key_hash = self._hash_key(value)

                scopes_env = os.getenv(f"API_KEY_{name}_SCOPES", "read")
                scopes = [s.strip() for s in scopes_env.split(",")]

                team = os.getenv(f"API_KEY_{name}_TEAM")
                rate_limit = int(os.getenv(f"API_KEY_{name}_RATE_LIMIT", "100"))

                self._api_keys[key_hash] = APIKey(
                    key_hash=key_hash,
                    name=name,
                    team=team,
                    scopes=scopes,
                    rate_limit=rate_limit,
                    enabled=True,
                )
                logger.info(f"Loaded API key: {name} (scopes: {scopes})")

        # Load master key if set (for admin access)
        master_key = os.getenv("API_MASTER_KEY")
        if master_key:
            key_hash = self._hash_key(master_key)
            self._api_keys[key_hash] = APIKey(
                key_hash=key_hash,
                name="master",
                scopes=["read", "write", "admin"],
                rate_limit=1000,
                enabled=True,
            )
            logger.info("Master API key configured")

        logger.info(f"API authentication enabled with {len(self._api_keys)} keys")

    def _hash_key(self, key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _extract_key(self) -> Optional[str]:
        """Extract API key from request."""
        # Check header first
        key = request.headers.get(self.HEADER_NAME)
        if key:
            return key

        # Check Authorization header for Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith(self.BEARER_PREFIX):
            return auth_header[len(self.BEARER_PREFIX) :]

        # Check query parameter
        key = request.args.get(self.QUERY_PARAM)
        if key:
            return key

        return None

    def _check_rate_limit(self, key_hash: str, limit: int) -> tuple[bool, int]:
        """
        Check if request is within rate limit.

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        window = 60  # 1 minute window

        if key_hash not in self._rate_limits:
            self._rate_limits[key_hash] = RateLimitState()

        state = self._rate_limits[key_hash]

        # Remove old requests outside the window
        state.requests = [t for t in state.requests if now - t < window]

        if len(state.requests) >= limit:
            return False, 0

        state.requests.append(now)
        return True, limit - len(state.requests)

    def validate_request(
        self, required_scopes: Optional[list[str]] = None
    ) -> tuple[bool, Optional[str], Optional[APIKey]]:
        """
        Validate the current request.

        Args:
            required_scopes: List of required scopes for this endpoint

        Returns:
            Tuple of (valid, error_message, api_key)
        """
        if not self._enabled:
            return True, None, None

        key = self._extract_key()
        if not key:
            return False, "API key required", None

        key_hash = self._hash_key(key)
        api_key = self._api_keys.get(key_hash)

        if not api_key:
            return False, "Invalid API key", None

        if not api_key.enabled:
            return False, "API key is disabled", None

        # Check scopes
        if required_scopes:
            missing_scopes = set(required_scopes) - set(api_key.scopes)
            if missing_scopes:
                return False, f"Missing required scopes: {missing_scopes}", None

        # Check rate limit
        allowed, remaining = self._check_rate_limit(key_hash, api_key.rate_limit)
        if not allowed:
            return False, "Rate limit exceeded", None

        return True, None, api_key

    @property
    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._enabled


# Global auth manager instance
auth_manager = AuthManager()


def require_auth(scopes: Optional[list[str]] = None):
    """
    Decorator to require API authentication for an endpoint.

    Args:
        scopes: List of required scopes (e.g., ["read"], ["write", "admin"])

    Usage:
        @app.route("/api/v1/protected")
        @require_auth(scopes=["read"])
        def protected_endpoint():
            return jsonify({"message": "authenticated"})
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            valid, error, api_key = auth_manager.validate_request(scopes)

            if not valid:
                return (
                    jsonify(
                        {
                            "error": "Authentication failed",
                            "message": error,
                        }
                    ),
                    401 if "Invalid" in error or "required" in error else 403,
                )

            # Add API key info to request context if authenticated
            if api_key:
                request.api_key = api_key
                request.api_key_name = api_key.name
                request.api_key_team = api_key.team

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def optional_auth():
    """
    Decorator that validates auth if provided but doesn't require it.

    Useful for endpoints that provide additional data to authenticated users.
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = request.headers.get(AuthManager.HEADER_NAME) or request.args.get(
                AuthManager.QUERY_PARAM
            )

            if key:
                valid, error, api_key = auth_manager.validate_request()
                if not valid:
                    return (
                        jsonify(
                            {
                                "error": "Authentication failed",
                                "message": error,
                            }
                        ),
                        401,
                    )

                if api_key:
                    request.api_key = api_key
                    request.api_key_name = api_key.name
                    request.api_key_team = api_key.team

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def initialize_auth():
    """Initialize authentication from environment."""
    auth_manager.initialize()