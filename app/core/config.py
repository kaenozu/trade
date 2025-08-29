"""Application configuration management."""

import os

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "JP Stocks ML Forecaster"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # CORS
    cors_origins: str = "*"
    
    # Monitoring & Observability
    metrics_enabled: bool = True
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.0
    sentry_profiles_sample_rate: float = 0.0
    sentry_env: str = "production"
    
    # Data & Cache
    cache_dir: str | None = None
    allow_synthetic_data: bool = False
    
    # Model
    model_dir: str | None = None
    
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
    def git_sha(self) -> str | None:
        """Get git SHA from environment."""
        return os.environ.get("GIT_SHA")


settings = Settings()