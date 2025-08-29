"""Application configuration management."""

import os
from enum import Enum
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment
    environment: Environment = Environment.DEVELOPMENT

    # Application
    app_name: str = "JP Stocks ML Forecaster"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = ""
    api_keys: str = ""  # Comma-separated API keys

    # CORS
    cors_origins: str = "*"
    cors_allow_credentials: bool = True

    # Security
    security_headers_enabled: bool = True
    rate_limiting_enabled: bool = True

    # Monitoring & Observability
    metrics_enabled: bool = True
    sentry_dsn: Optional[str] = None
    sentry_traces_sample_rate: float = 0.0
    sentry_profiles_sample_rate: float = 0.0
    sentry_env: str = "production"

    # Data & Cache
    cache_dir: Optional[str] = None
    cache_ttl_seconds: int = 3600  # 1 hour default
    allow_synthetic_data: bool = False

    # Model
    model_dir: Optional[str] = None
    model_cache_enabled: bool = True

    # Performance
    max_workers: int = 4
    request_timeout_seconds: int = 30

    # Feature flags
    async_data_fetch: bool = True
    enhanced_caching: bool = True
    domain_validation: bool = True

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "case_sensitive": False,
    }

    @property
    def cache_directory(self) -> str:
        """Get cache directory path."""
        return self.cache_dir or os.path.join(os.getcwd(), "cache")

    @property
    def model_directory(self) -> str:
        """Get model directory path."""
        return self.model_dir or os.path.join(os.getcwd(), "models")

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def git_sha(self) -> Optional[str]:
        """Get git SHA from environment."""
        return os.environ.get("GIT_SHA")

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def get_database_url(self) -> str:
        """Get database URL based on environment."""
        # For future database integration
        if self.is_production:
            return os.environ.get("DATABASE_URL", "postgresql://prod_db")
        elif self.environment == Environment.STAGING:
            return os.environ.get("DATABASE_URL", "postgresql://staging_db")
        else:
            return os.environ.get("DATABASE_URL", "sqlite:///./test.db")

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL for caching if available."""
        return os.environ.get("REDIS_URL")

    def get_log_config(self) -> dict:
        """Get logging configuration based on environment."""
        base_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": self.log_level,
                "handlers": ["default"],
            },
        }

        if self.is_production:
            # In production, use detailed formatter and consider file logging
            base_config["handlers"]["default"]["formatter"] = "detailed"

            # Add file handler if log file is configured
            log_file = os.environ.get("LOG_FILE")
            if log_file:
                base_config["handlers"]["file"] = {
                    "formatter": "detailed",
                    "class": "logging.FileHandler",
                    "filename": log_file,
                }
                base_config["root"]["handlers"].append("file")

        return base_config

    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        if self.is_production:
            if not self.sentry_dsn:
                issues.append("Sentry DSN should be configured in production")

            if self.debug:
                issues.append("Debug mode should be disabled in production")

            if self.cors_origins == "*":
                issues.append("CORS should be restricted in production")

            if not self.api_keys:
                issues.append("API keys should be configured in production")

        if self.cache_ttl_seconds <= 0:
            issues.append("Cache TTL must be positive")

        if self.max_workers <= 0:
            issues.append("Max workers must be positive")

        return issues


def get_settings() -> Settings:
    """Get settings instance with validation."""
    settings = Settings()

    # Validate configuration
    issues = settings.validate_config()
    if issues:
        print(f"Configuration issues found: {'; '.join(issues)}")
        if settings.is_production:
            raise ValueError(f"Critical configuration issues in production: {'; '.join(issues)}")

    return settings


# Global settings instance
settings = get_settings()
