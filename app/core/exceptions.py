"""Custom exceptions for the application."""

from typing import Any


class BaseAppException(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class DataError(BaseAppException):
    """Data-related errors (fetching, processing)."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "DATA_ERROR", details)


class ModelError(BaseAppException):
    """ML model-related errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "MODEL_ERROR", details)


class ValidationError(BaseAppException):
    """Input validation errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class ConfigurationError(BaseAppException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)