"""Security utilities and validation enhancements."""

import hashlib
import hmac
import re
import secrets
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED

from .config import settings


class InputSanitizer:
    """Utility class for input sanitization and validation."""

    # Common dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',  # Iframe tags
        r'<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>',  # Object tags
        r'<embed\b[^<]*(?:(?!<\/embed>)<[^<]*)*<\/embed>',  # Embed tags
    ]

    # SQL injection patterns
    SQL_PATTERNS = [
        r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
        r'--',  # SQL comments
        r'/\*.*?\*/',  # Multi-line comments
        r"'.*?'",  # String literals (potential SQL)
    ]

    def __init__(self):
        self.dangerous_regex = re.compile('|'.join(self.DANGEROUS_PATTERNS), re.IGNORECASE)
        self.sql_regex = re.compile('|'.join(self.SQL_PATTERNS), re.IGNORECASE)

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            raise ValueError("Value must be a string")

        # Limit length
        if len(value) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")

        # Remove dangerous patterns
        if self.dangerous_regex.search(value):
            raise ValueError("Input contains potentially dangerous content")

        # Check for SQL injection attempts
        if self.sql_regex.search(value):
            raise ValueError("Input contains potential SQL injection patterns")

        # Basic sanitization
        sanitized = value.strip()

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        return sanitized

    def validate_ticker(self, ticker: str) -> str:
        """Validate and sanitize a ticker symbol."""
        if not ticker:
            raise ValueError("Ticker cannot be empty")

        sanitized = self.sanitize_string(ticker, max_length=15)

        # Ticker-specific validation
        if not re.match(r'^[A-Za-z0-9._-]+$', sanitized):
            raise ValueError("Invalid ticker format")

        return sanitized.upper()

    def validate_numeric_param(
        self,
        value: Any,
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """Validate a numeric parameter."""
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{param_name} must be a valid number")

        if not (-1e10 <= numeric_value <= 1e10):  # Reasonable bounds
            raise ValueError(f"{param_name} is out of reasonable range")

        if min_value is not None and numeric_value < min_value:
            raise ValueError(f"{param_name} must be >= {min_value}")

        if max_value is not None and numeric_value > max_value:
            raise ValueError(f"{param_name} must be <= {max_value}")

        return numeric_value


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""

    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()

    def is_allowed(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 3600
    ) -> bool:
        """Check if a request is allowed under rate limiting rules."""
        current_time = time.time()

        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time - window_seconds)
            self.last_cleanup = current_time

        # Get existing requests for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []

        request_times = self.requests[identifier]

        # Remove old requests outside the window
        cutoff_time = current_time - window_seconds
        request_times[:] = [t for t in request_times if t > cutoff_time]

        # Check if under limit
        if len(request_times) >= max_requests:
            return False

        # Add current request
        request_times.append(current_time)
        return True

    def _cleanup_old_requests(self, cutoff_time: float):
        """Remove old request records to prevent memory leaks."""
        identifiers_to_remove = []

        for identifier, request_times in self.requests.items():
            # Remove old requests
            request_times[:] = [t for t in request_times if t > cutoff_time]

            # Remove empty entries
            if not request_times:
                identifiers_to_remove.append(identifier)

        for identifier in identifiers_to_remove:
            del self.requests[identifier]


class APIKeyValidator:
    """Validator for API keys if authentication is enabled."""

    def __init__(self):
        self.valid_keys: Set[str] = set()
        self.key_metadata: Dict[str, dict] = {}

        # Load API keys from environment if available
        api_keys_env = getattr(settings, 'api_keys', '')
        if api_keys_env:
            for key in api_keys_env.split(','):
                key = key.strip()
                if key:
                    self.add_api_key(key)

    def add_api_key(self, key: str, metadata: Optional[dict] = None):
        """Add a valid API key."""
        hashed_key = self._hash_key(key)
        self.valid_keys.add(hashed_key)
        self.key_metadata[hashed_key] = metadata or {}

    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        if not key:
            return False

        hashed_key = self._hash_key(key)
        return hashed_key in self.valid_keys

    def get_key_metadata(self, key: str) -> dict:
        """Get metadata for an API key."""
        hashed_key = self._hash_key(key)
        return self.key_metadata.get(hashed_key, {})

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage using secure key derivation."""
        # Use PBKDF2 for secure key hashing instead of SHA256
        salt = b'api_key_salt_v1'  # In production, use random salt per key
        return hashlib.pbkdf2_hmac('sha256', key.encode(), salt, 100000).hex()


# Global instances
_sanitizer = InputSanitizer()
_rate_limiter = RateLimiter()
_api_key_validator = APIKeyValidator()


def get_sanitizer() -> InputSanitizer:
    """Get the global input sanitizer."""
    return _sanitizer


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    return _rate_limiter


def get_api_key_validator() -> APIKeyValidator:
    """Get the global API key validator."""
    return _api_key_validator


def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Decorator for rate limiting endpoints."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract Request from args/kwargs robustly
            req_obj = None
            for a in args:
                if isinstance(a, Request):
                    req_obj = a
                    break
            if req_obj is None:
                for k in ("request", "http_request"):
                    v = kwargs.get(k)
                    if isinstance(v, Request):
                        req_obj = v
                        break

            client_ip = req_obj.client.host if (req_obj and req_obj.client) else "unknown"

            # Check rate limit
            if not get_rate_limiter().is_allowed(client_ip, max_requests, window_seconds):
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: max {max_requests} requests per {window_seconds} seconds"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(**validators):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            sanitizer = get_sanitizer()

            # Apply validators to kwargs
            for param_name, validator_config in validators.items():
                if param_name in kwargs:
                    value = kwargs[param_name]

                    if validator_config.get('type') == 'ticker':
                        kwargs[param_name] = sanitizer.validate_ticker(value)

                    elif validator_config.get('type') == 'numeric':
                        kwargs[param_name] = sanitizer.validate_numeric_param(
                            value,
                            param_name,
                            validator_config.get('min'),
                            validator_config.get('max')
                        )

                    elif validator_config.get('type') == 'string':
                        kwargs[param_name] = sanitizer.sanitize_string(
                            value,
                            validator_config.get('max_length', 1000)
                        )

                try:
                    return await func(*args, **kwargs)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
        return wrapper
    return decorator


class SecurityHeadersMiddleware:
    """Middleware to add security headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Add security headers
                security_headers = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"x-xss-protection": b"1; mode=block",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"geolocation=(), microphone=(), camera=()",
                }

                # Add CSP if not in debug mode
                if not settings.debug:
                    security_headers[b"content-security-policy"] = (
                        b"default-src 'self'; "
                        b"script-src 'self' 'unsafe-inline'; "
                        b"style-src 'self' 'unsafe-inline'; "
                        b"img-src 'self' data:; "
                        b"font-src 'self'; "
                        b"connect-src 'self'"
                    )

                # Merge with existing headers
                for key, value in security_headers.items():
                    headers[key] = value

                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_with_security_headers)


def setup_security_headers(app):
    """Setup security headers middleware."""
    app.add_middleware(SecurityHeadersMiddleware)
