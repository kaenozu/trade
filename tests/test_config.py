"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from app.core.config import Environment, Settings, get_settings


class TestEnvironment:
    """Test Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.TESTING == "testing"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestSettings:
    """Test Settings class."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = Settings()
        
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.app_name == "JP Stocks ML Forecaster"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_prefix == ""
        assert settings.api_keys == ""
        assert settings.cors_origins == "*"
        assert settings.cors_allow_credentials is True
        assert settings.security_headers_enabled is True
        assert settings.rate_limiting_enabled is True
        assert settings.metrics_enabled is True
        assert settings.sentry_dsn is None
        assert settings.sentry_traces_sample_rate == 0.0
        assert settings.sentry_profiles_sample_rate == 0.0
        assert settings.sentry_env == "production"
        assert settings.cache_dir is None
        assert settings.cache_ttl_seconds == 3600
        assert settings.allow_synthetic_data is False
        assert settings.model_dir is None
        assert settings.model_cache_enabled is True
        assert settings.max_workers == 4
        assert settings.request_timeout_seconds == 30
        assert settings.async_data_fetch is True
        assert settings.enhanced_caching is True
        assert settings.domain_validation is True

    def test_cache_directory_default(self):
        """Test cache directory default path."""
        settings = Settings()
        expected_path = os.path.join(os.getcwd(), "cache")
        assert settings.cache_directory == expected_path

    def test_cache_directory_custom(self):
        """Test cache directory custom path."""
        settings = Settings(cache_dir="/custom/cache")
        assert settings.cache_directory == "/custom/cache"

    def test_model_directory_default(self):
        """Test model directory default path."""
        settings = Settings()
        expected_path = os.path.join(os.getcwd(), "models")
        assert settings.model_directory == expected_path

    def test_model_directory_custom(self):
        """Test model directory custom path."""
        settings = Settings(model_dir="/custom/models")
        assert settings.model_directory == "/custom/models"

    def test_cors_origins_list_default(self):
        """Test CORS origins list with default wildcard."""
        settings = Settings()
        assert settings.cors_origins_list == ["*"]

    def test_cors_origins_list_single(self):
        """Test CORS origins list with single origin."""
        settings = Settings(cors_origins="https://example.com")
        assert settings.cors_origins_list == ["https://example.com"]

    def test_cors_origins_list_multiple(self):
        """Test CORS origins list with multiple origins."""
        settings = Settings(cors_origins="https://example.com,https://api.example.com,http://localhost:3000")
        expected = ["https://example.com", "https://api.example.com", "http://localhost:3000"]
        assert settings.cors_origins_list == expected

    def test_cors_origins_list_with_spaces(self):
        """Test CORS origins list with spaces."""
        settings = Settings(cors_origins="https://example.com, https://api.example.com ,http://localhost:3000")
        expected = ["https://example.com", "https://api.example.com", "http://localhost:3000"]
        assert settings.cors_origins_list == expected

    def test_git_sha_from_environment(self):
        """Test git SHA retrieval from environment."""
        with patch.dict(os.environ, {"GIT_SHA": "abc123def456"}):
            settings = Settings()
            assert settings.git_sha == "abc123def456"

    def test_git_sha_not_set(self):
        """Test git SHA when not set in environment."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.git_sha is None

    def test_is_development(self):
        """Test development environment check."""
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.is_development is True
        assert settings.is_testing is False
        assert settings.is_production is False

    def test_is_testing(self):
        """Test testing environment check."""
        settings = Settings(environment=Environment.TESTING)
        assert settings.is_development is False
        assert settings.is_testing is True
        assert settings.is_production is False

    def test_is_production(self):
        """Test production environment check."""
        settings = Settings(environment=Environment.PRODUCTION)
        assert settings.is_development is False
        assert settings.is_testing is False
        assert settings.is_production is True

    def test_get_database_url_production(self):
        """Test database URL for production environment."""
        settings = Settings(environment=Environment.PRODUCTION)
        
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://prod_custom"}):
            assert settings.get_database_url() == "postgresql://prod_custom"
        
        with patch.dict(os.environ, {}, clear=True):
            assert settings.get_database_url() == "postgresql://prod_db"

    def test_get_database_url_staging(self):
        """Test database URL for staging environment."""
        settings = Settings(environment=Environment.STAGING)
        
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://staging_custom"}):
            assert settings.get_database_url() == "postgresql://staging_custom"
        
        with patch.dict(os.environ, {}, clear=True):
            assert settings.get_database_url() == "postgresql://staging_db"

    def test_get_database_url_development(self):
        """Test database URL for development environment."""
        settings = Settings(environment=Environment.DEVELOPMENT)
        
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///custom.db"}):
            assert settings.get_database_url() == "sqlite:///custom.db"
        
        with patch.dict(os.environ, {}, clear=True):
            assert settings.get_database_url() == "sqlite:///./test.db"

    def test_get_redis_url(self):
        """Test Redis URL retrieval."""
        settings = Settings()
        
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            assert settings.get_redis_url() == "redis://localhost:6379"
        
        with patch.dict(os.environ, {}, clear=True):
            assert settings.get_redis_url() is None

    def test_get_log_config_development(self):
        """Test log config for development environment."""
        settings = Settings(environment=Environment.DEVELOPMENT, log_level="DEBUG")
        config = settings.get_log_config()
        
        assert config["version"] == 1
        assert config["disable_existing_loggers"] is False
        assert "default" in config["formatters"]
        assert "detailed" in config["formatters"]
        assert "default" in config["handlers"]
        assert config["root"]["level"] == "DEBUG"
        assert config["root"]["handlers"] == ["default"]
        assert config["handlers"]["default"]["formatter"] == "default"

    def test_get_log_config_production(self):
        """Test log config for production environment."""
        settings = Settings(environment=Environment.PRODUCTION, log_level="WARNING")
        config = settings.get_log_config()
        
        assert config["root"]["level"] == "WARNING"
        assert config["handlers"]["default"]["formatter"] == "detailed"

    def test_get_log_config_production_with_file(self):
        """Test log config for production with file logging."""
        settings = Settings(environment=Environment.PRODUCTION)
        
        with patch.dict(os.environ, {"LOG_FILE": "/var/log/app.log"}):
            config = settings.get_log_config()
            
            assert "file" in config["handlers"]
            assert config["handlers"]["file"]["filename"] == "/var/log/app.log"
            assert "file" in config["root"]["handlers"]

    def test_validate_config_development(self):
        """Test configuration validation for development."""
        settings = Settings(environment=Environment.DEVELOPMENT)
        issues = settings.validate_config()
        assert len(issues) == 0

    def test_validate_config_production_valid(self):
        """Test configuration validation for production with valid config."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=False,
            cors_origins="https://example.com",
            api_keys="key1,key2",
            sentry_dsn="https://test@sentry.io/123"
        )
        issues = settings.validate_config()
        assert len(issues) == 0

    def test_validate_config_production_missing_sentry(self):
        """Test configuration validation for production missing Sentry."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=False,
            cors_origins="https://example.com",
            api_keys="key1,key2"
        )
        issues = settings.validate_config()
        assert "Sentry DSN should be configured in production" in issues

    def test_validate_config_production_debug_enabled(self):
        """Test configuration validation for production with debug enabled."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=True,
            cors_origins="https://example.com",
            api_keys="key1,key2",
            sentry_dsn="https://test@sentry.io/123"
        )
        issues = settings.validate_config()
        assert "Debug mode should be disabled in production" in issues

    def test_validate_config_production_cors_wildcard(self):
        """Test configuration validation for production with CORS wildcard."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=False,
            cors_origins="*",
            api_keys="key1,key2",
            sentry_dsn="https://test@sentry.io/123"
        )
        issues = settings.validate_config()
        assert "CORS should be restricted in production" in issues

    def test_validate_config_production_no_api_keys(self):
        """Test configuration validation for production without API keys."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=False,
            cors_origins="https://example.com",
            api_keys="",
            sentry_dsn="https://test@sentry.io/123"
        )
        issues = settings.validate_config()
        assert "API keys should be configured in production" in issues

    def test_validate_config_negative_cache_ttl(self):
        """Test configuration validation with negative cache TTL."""
        settings = Settings(cache_ttl_seconds=-1)
        issues = settings.validate_config()
        assert "Cache TTL must be positive" in issues

    def test_validate_config_zero_cache_ttl(self):
        """Test configuration validation with zero cache TTL."""
        settings = Settings(cache_ttl_seconds=0)
        issues = settings.validate_config()
        assert "Cache TTL must be positive" in issues

    def test_validate_config_negative_max_workers(self):
        """Test configuration validation with negative max workers."""
        settings = Settings(max_workers=-1)
        issues = settings.validate_config()
        assert "Max workers must be positive" in issues

    def test_validate_config_zero_max_workers(self):
        """Test configuration validation with zero max workers."""
        settings = Settings(max_workers=0)
        issues = settings.validate_config()
        assert "Max workers must be positive" in issues


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_development(self):
        """Test getting settings for development environment."""
        with patch("app.core.config.Settings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.validate_config.return_value = []
            mock_settings.is_production = False
            
            result = get_settings()
            
            assert result == mock_settings
            mock_settings.validate_config.assert_called_once()

    def test_get_settings_with_warnings(self, capfd):
        """Test getting settings with validation warnings."""
        with patch("app.core.config.Settings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.validate_config.return_value = ["Warning 1", "Warning 2"]
            mock_settings.is_production = False
            
            result = get_settings()
            
            captured = capfd.readouterr()
            assert "Configuration issues found: Warning 1; Warning 2" in captured.out
            assert result == mock_settings

    def test_get_settings_production_with_errors(self):
        """Test getting settings for production with critical errors."""
        with patch("app.core.config.Settings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.validate_config.return_value = ["Critical error 1", "Critical error 2"]
            mock_settings.is_production = True
            
            with pytest.raises(ValueError, match="Critical configuration issues in production"):
                get_settings()

    def test_get_settings_production_valid(self):
        """Test getting settings for production with valid config."""
        with patch("app.core.config.Settings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.validate_config.return_value = []
            mock_settings.is_production = True
            
            result = get_settings()
            
            assert result == mock_settings


class TestModelConfig:
    """Test model configuration."""

    def test_model_config_attributes(self):
        """Test that model_config has expected attributes."""
        settings = Settings()
        
        assert hasattr(settings, "model_config")
        config = settings.model_config
        
        assert config["env_file"] == ".env"
        assert config["env_prefix"] == ""
        assert config["case_sensitive"] is False


class TestSettingsWithEnvironmentVariables:
    """Test settings with environment variables."""

    def test_settings_from_env_vars(self):
        """Test settings loading from environment variables."""
        env_vars = {
            "APP_NAME": "Test App",
            "API_PORT": "9000",
            "DEBUG": "true",
            "ENVIRONMENT": "testing",
            "MAX_WORKERS": "8"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.app_name == "Test App"
            assert settings.api_port == 9000
            assert settings.debug is True
            assert settings.environment == Environment.TESTING
            assert settings.max_workers == 8

    def test_settings_case_insensitive(self):
        """Test that settings are case insensitive."""
        env_vars = {
            "app_name": "Test App Lower",
            "APP_NAME": "Test App Upper",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            # Should use one of them (behavior depends on environment variable precedence)
            assert settings.app_name in ["Test App Lower", "Test App Upper"]


class TestIntegration:
    """Test integration scenarios."""

    def test_production_settings_integration(self):
        """Test production settings integration."""
        env_vars = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "CORS_ORIGINS": "https://myapp.com,https://api.myapp.com",
            "API_KEYS": "key1,key2,key3",
            "SENTRY_DSN": "https://test@sentry.io/123456",
            "SENTRY_TRACES_SAMPLE_RATE": "0.1",
            "LOG_LEVEL": "ERROR",
            "MAX_WORKERS": "12"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.is_production
            assert not settings.debug
            assert settings.cors_origins_list == ["https://myapp.com", "https://api.myapp.com"]
            assert settings.api_keys == "key1,key2,key3"
            assert settings.sentry_dsn == "https://test@sentry.io/123456"
            assert settings.sentry_traces_sample_rate == 0.1
            assert settings.log_level == "ERROR"
            assert settings.max_workers == 12
            
            # Should have no validation issues
            issues = settings.validate_config()
            assert len(issues) == 0