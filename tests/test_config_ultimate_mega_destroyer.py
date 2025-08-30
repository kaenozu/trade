"""最強の Configuration Management 超究極メガデストロイヤーテスト - 212行の完全制覇！"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from app.core.config import (
    Environment, Settings, get_settings
)


class TestEnvironmentEnum:
    """Environment Enumの完全制覇テスト"""

    def test_environment_values_destroyer(self):
        """Environment Enum値の完全制覇"""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        
        # Enumとしての動作確認
        assert isinstance(Environment.PRODUCTION, Environment)
        assert Environment.DEVELOPMENT == "development"

    def test_environment_comparison_destroyer(self):
        """Environment比較演算の完全制覇"""
        assert Environment.DEVELOPMENT != Environment.PRODUCTION
        assert Environment.TESTING == "testing"
        assert Environment.STAGING in [Environment.STAGING, Environment.PRODUCTION]


class TestSettingsDefaults:
    """Settings デフォルト値の完全制覇テスト"""

    def test_default_settings_destroyer(self):
        """デフォルト設定の完全制覇"""
        settings = Settings()
        
        # Environment
        assert settings.environment == Environment.DEVELOPMENT
        
        # Application
        assert settings.app_name == "JP Stocks ML Forecaster"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        
        # API
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_prefix == ""
        assert settings.api_keys == ""
        
        # CORS
        assert settings.cors_origins == "*"
        assert settings.cors_allow_credentials is True
        
        # Security
        assert settings.security_headers_enabled is True
        assert settings.rate_limiting_enabled is True
        
        # Monitoring
        assert settings.metrics_enabled is True
        assert settings.sentry_dsn is None
        assert settings.sentry_traces_sample_rate == 0.0
        assert settings.sentry_profiles_sample_rate == 0.0
        assert settings.sentry_env == "production"
        
        # Data & Cache
        assert settings.cache_dir is None
        assert settings.cache_ttl_seconds == 3600
        assert settings.allow_synthetic_data is False
        
        # Model
        assert settings.model_dir is None
        assert settings.model_cache_enabled is True
        
        # Performance
        assert settings.max_workers == 4
        assert settings.request_timeout_seconds == 30
        
        # Feature flags
        assert settings.async_data_fetch is True
        assert settings.enhanced_caching is True
        assert settings.domain_validation is True

    def test_model_config_destroyer(self):
        """モデル設定の完全制覇"""
        settings = Settings()
        
        assert hasattr(settings, 'model_config')
        config = settings.model_config
        assert config['env_file'] == '.env'
        assert config['env_prefix'] == ''
        assert config['case_sensitive'] is False


class TestSettingsProperties:
    """Settings プロパティの完全制覇テスト"""

    def test_cache_directory_property_destroyer(self):
        """cache_directoryプロパティの完全制覇"""
        # デフォルトの場合
        settings = Settings()
        expected = os.path.join(os.getcwd(), "cache")
        assert settings.cache_directory == expected
        
        # カスタム設定の場合
        custom_dir = "/custom/cache"
        settings = Settings(cache_dir=custom_dir)
        assert settings.cache_directory == custom_dir

    def test_model_directory_property_destroyer(self):
        """model_directoryプロパティの完全制覇"""
        # デフォルトの場合
        settings = Settings()
        expected = os.path.join(os.getcwd(), "models")
        assert settings.model_directory == expected
        
        # カスタム設定の場合
        custom_dir = "/custom/models"
        settings = Settings(model_dir=custom_dir)
        assert settings.model_directory == custom_dir

    def test_cors_origins_list_property_destroyer(self):
        """cors_origins_listプロパティの完全制覇"""
        # ワイルドカードの場合
        settings = Settings(cors_origins="*")
        assert settings.cors_origins_list == ["*"]
        
        # 単一オリジンの場合
        settings = Settings(cors_origins="https://example.com")
        assert settings.cors_origins_list == ["https://example.com"]
        
        # 複数オリジンの場合
        settings = Settings(cors_origins="https://app.com, https://api.com, http://localhost:3000")
        expected = ["https://app.com", "https://api.com", "http://localhost:3000"]
        assert settings.cors_origins_list == expected
        
        # 空白含みの場合
        settings = Settings(cors_origins=" https://test.com , https://dev.com ")
        assert settings.cors_origins_list == ["https://test.com", "https://dev.com"]

    @patch.dict(os.environ, {"GIT_SHA": "abc123def456"})
    def test_git_sha_property_destroyer(self):
        """git_shaプロパティの完全制覇"""
        settings = Settings()
        assert settings.git_sha == "abc123def456"
        
    @patch.dict(os.environ, {}, clear=True)
    def test_git_sha_property_none_destroyer(self):
        """git_sha未設定の完全制覇"""
        settings = Settings()
        assert settings.git_sha is None

    def test_environment_check_properties_destroyer(self):
        """環境チェックプロパティの完全制覇"""
        # Development
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.is_development is True
        assert settings.is_testing is False
        assert settings.is_production is False
        
        # Testing
        settings = Settings(environment=Environment.TESTING)
        assert settings.is_development is False
        assert settings.is_testing is True
        assert settings.is_production is False
        
        # Production
        settings = Settings(environment=Environment.PRODUCTION)
        assert settings.is_development is False
        assert settings.is_testing is False
        assert settings.is_production is True
        
        # Staging
        settings = Settings(environment=Environment.STAGING)
        assert settings.is_development is False
        assert settings.is_testing is False
        assert settings.is_production is False


class TestDatabaseUrl:
    """Database URL取得の完全制覇テスト"""

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://prod_custom"})
    def test_get_database_url_production_custom_destroyer(self):
        """本番環境カスタムURL の完全制覇"""
        settings = Settings(environment=Environment.PRODUCTION)
        assert settings.get_database_url() == "postgresql://prod_custom"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_url_production_default_destroyer(self):
        """本番環境デフォルトURL の完全制覇"""
        settings = Settings(environment=Environment.PRODUCTION)
        assert settings.get_database_url() == "postgresql://prod_db"

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://staging_custom"})
    def test_get_database_url_staging_custom_destroyer(self):
        """ステージング環境カスタムURL の完全制覇"""
        settings = Settings(environment=Environment.STAGING)
        assert settings.get_database_url() == "postgresql://staging_custom"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_url_staging_default_destroyer(self):
        """ステージング環境デフォルトURL の完全制覇"""
        settings = Settings(environment=Environment.STAGING)
        assert settings.get_database_url() == "postgresql://staging_db"

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://dev_custom"})
    def test_get_database_url_development_custom_destroyer(self):
        """開発環境カスタムURL の完全制覇"""
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.get_database_url() == "postgresql://dev_custom"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_url_development_default_destroyer(self):
        """開発環境デフォルトURL の完全制覇"""
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.get_database_url() == "sqlite:///./test.db"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_url_testing_default_destroyer(self):
        """テスト環境デフォルトURL の完全制覇"""
        settings = Settings(environment=Environment.TESTING)
        assert settings.get_database_url() == "sqlite:///./test.db"


class TestRedisUrl:
    """Redis URL取得の完全制覇テスト"""

    @patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"})
    def test_get_redis_url_configured_destroyer(self):
        """Redis URL設定済みの完全制覇"""
        settings = Settings()
        assert settings.get_redis_url() == "redis://localhost:6379/0"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_redis_url_not_configured_destroyer(self):
        """Redis URL未設定の完全制覇"""
        settings = Settings()
        assert settings.get_redis_url() is None


class TestLogConfig:
    """ログ設定の完全制覇テスト"""

    def test_get_log_config_development_destroyer(self):
        """開発環境ログ設定の完全制覇"""
        settings = Settings(environment=Environment.DEVELOPMENT, log_level="DEBUG")
        config = settings.get_log_config()
        
        assert config["version"] == 1
        assert config["disable_existing_loggers"] is False
        assert "formatters" in config
        assert "default" in config["formatters"]
        assert "detailed" in config["formatters"]
        assert config["handlers"]["default"]["formatter"] == "default"
        assert config["root"]["level"] == "DEBUG"
        assert config["root"]["handlers"] == ["default"]

    @patch.dict(os.environ, {}, clear=True)
    def test_get_log_config_production_no_file_destroyer(self):
        """本番環境ファイルログなし設定の完全制覇"""
        settings = Settings(environment=Environment.PRODUCTION, log_level="WARNING")
        config = settings.get_log_config()
        
        assert config["handlers"]["default"]["formatter"] == "detailed"
        assert config["root"]["level"] == "WARNING"
        assert config["root"]["handlers"] == ["default"]
        assert "file" not in config["handlers"]

    @patch.dict(os.environ, {"LOG_FILE": "/var/log/app.log"})
    def test_get_log_config_production_with_file_destroyer(self):
        """本番環境ファイルログあり設定の完全制覇"""
        settings = Settings(environment=Environment.PRODUCTION, log_level="ERROR")
        config = settings.get_log_config()
        
        assert config["handlers"]["default"]["formatter"] == "detailed"
        assert "file" in config["handlers"]
        assert config["handlers"]["file"]["filename"] == "/var/log/app.log"
        assert config["handlers"]["file"]["formatter"] == "detailed"
        assert config["root"]["handlers"] == ["default", "file"]


class TestConfigValidation:
    """設定検証の完全制覇テスト"""

    def test_validate_config_development_destroyer(self):
        """開発環境検証の完全制覇"""
        settings = Settings(environment=Environment.DEVELOPMENT)
        issues = settings.validate_config()
        # 開発環境では特に問題なし
        assert len(issues) == 0

    def test_validate_config_production_minimal_destroyer(self):
        """本番環境最小構成検証の完全制覇"""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=False,
            cors_origins="https://myapp.com",
            api_keys="key1,key2",
            sentry_dsn="https://sentry.io/project"
        )
        issues = settings.validate_config()
        assert len(issues) == 0

    def test_validate_config_production_issues_destroyer(self):
        """本番環境問題あり検証の完全制覇"""
        settings = Settings(
            environment=Environment.PRODUCTION,
            debug=True,  # 問題: デバッグが有効
            cors_origins="*",  # 問題: CORS制限なし
            api_keys="",  # 問題: APIキーなし
            sentry_dsn=None  # 問題: Sentryなし
        )
        issues = settings.validate_config()
        
        assert len(issues) == 4
        assert "Debug mode should be disabled in production" in issues
        assert "CORS should be restricted in production" in issues
        assert "API keys should be configured in production" in issues
        assert "Sentry DSN should be configured in production" in issues

    def test_validate_config_cache_ttl_destroyer(self):
        """キャッシュTTL検証の完全制覇"""
        settings = Settings(cache_ttl_seconds=0)
        issues = settings.validate_config()
        assert "Cache TTL must be positive" in issues
        
        settings = Settings(cache_ttl_seconds=-100)
        issues = settings.validate_config()
        assert "Cache TTL must be positive" in issues
        
        settings = Settings(cache_ttl_seconds=1)
        issues = settings.validate_config()
        assert "Cache TTL must be positive" not in issues

    def test_validate_config_max_workers_destroyer(self):
        """最大ワーカー数検証の完全制覇"""
        settings = Settings(max_workers=0)
        issues = settings.validate_config()
        assert "Max workers must be positive" in issues
        
        settings = Settings(max_workers=-5)
        issues = settings.validate_config()
        assert "Max workers must be positive" in issues
        
        settings = Settings(max_workers=1)
        issues = settings.validate_config()
        assert "Max workers must be positive" not in issues


class TestGetSettings:
    """get_settings()関数の完全制覇テスト"""

    @patch('app.core.config.Settings')
    def test_get_settings_valid_destroyer(self, mock_settings_class):
        """有効設定でのget_settings完全制覇"""
        mock_settings = MagicMock()
        mock_settings.validate_config.return_value = []
        mock_settings.is_production = False
        mock_settings_class.return_value = mock_settings
        
        from app.core.config import get_settings
        result = get_settings()
        
        assert result == mock_settings
        mock_settings.validate_config.assert_called_once()

    @patch('app.core.config.Settings')
    @patch('builtins.print')
    def test_get_settings_issues_non_production_destroyer(self, mock_print, mock_settings_class):
        """非本番環境での問題ありget_settings完全制覇"""
        mock_settings = MagicMock()
        mock_settings.validate_config.return_value = ["Some warning"]
        mock_settings.is_production = False
        mock_settings_class.return_value = mock_settings
        
        from app.core.config import get_settings
        result = get_settings()
        
        assert result == mock_settings
        mock_print.assert_called_once_with("Configuration issues found: Some warning")

    @patch('app.core.config.Settings')
    def test_get_settings_issues_production_destroyer(self, mock_settings_class):
        """本番環境での問題ありget_settings完全制覇"""
        mock_settings = MagicMock()
        mock_settings.validate_config.return_value = ["Critical issue"]
        mock_settings.is_production = True
        mock_settings_class.return_value = mock_settings
        
        from app.core.config import get_settings
        with pytest.raises(ValueError, match="Critical configuration issues in production: Critical issue"):
            get_settings()


class TestEnvironmentVariableLoading:
    """環境変数読み込みの完全制覇テスト"""

    @patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "API_PORT": "9000",
        "LOG_LEVEL": "ERROR"
    })
    def test_environment_variable_loading_destroyer(self):
        """環境変数からの設定読み込み完全制覇"""
        settings = Settings()
        
        assert settings.environment == Environment.PRODUCTION
        assert settings.debug is False
        assert settings.api_port == 9000
        assert settings.log_level == "ERROR"

    @patch.dict(os.environ, {"METRICS_ENABLED": "false", "CACHE_TTL_SECONDS": "7200"})
    def test_boolean_and_numeric_env_vars_destroyer(self):
        """真偽値・数値環境変数の完全制覇"""
        settings = Settings()
        
        assert settings.metrics_enabled is False
        assert settings.cache_ttl_seconds == 7200


class TestEdgeCases:
    """エッジケース処理の完全制覇テスト"""

    def test_empty_cors_origins_destroyer(self):
        """空CORS設定の完全制覇"""
        settings = Settings(cors_origins="")
        assert settings.cors_origins_list == [""]

    def test_cors_origins_with_only_commas_destroyer(self):
        """コンマのみCORS設定の完全制覇"""
        settings = Settings(cors_origins=",,,")
        assert settings.cors_origins_list == ["", "", "", ""]

    def test_extreme_values_destroyer(self):
        """極端値設定の完全制覇"""
        settings = Settings(
            cache_ttl_seconds=999999,
            max_workers=1000,
            request_timeout_seconds=3600,
            sentry_traces_sample_rate=1.0,
            sentry_profiles_sample_rate=1.0
        )
        
        assert settings.cache_ttl_seconds == 999999
        assert settings.max_workers == 1000
        assert settings.request_timeout_seconds == 3600
        assert settings.sentry_traces_sample_rate == 1.0
        assert settings.sentry_profiles_sample_rate == 1.0


class TestGlobalSettingsInstance:
    """グローバル設定インスタンスの完全制覇テスト"""

    def test_global_settings_import_destroyer(self):
        """グローバル設定インポートの完全制覇"""
        from app.core.config import settings
        
        assert isinstance(settings, Settings)
        assert hasattr(settings, 'environment')
        assert hasattr(settings, 'app_name')

    def test_global_settings_creation_destroyer(self):
        """グローバル設定作成の完全制覇"""
        # インポート時にget_settingsが既に呼ばれており、
        # グローバルインスタンスが作成されていることを確認
        from app.core.config import settings
        
        # グローバルインスタンスが正しく初期化されている
        assert isinstance(settings, Settings)
        assert hasattr(settings, 'validate_config')
        assert callable(settings.validate_config)
        
        # 設定値がデフォルト値で初期化されている
        assert settings.app_name == "JP Stocks ML Forecaster"
        assert settings.environment == Environment.DEVELOPMENT
        
        # 再インポート後もSettingsインスタンス
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        
        from app.core.config import settings as reloaded_settings, Settings as ReloadedSettings
        assert isinstance(reloaded_settings, ReloadedSettings)


class TestConfigIntegration:
    """設定統合テストの完全制覇"""

    def test_full_production_config_destroyer(self):
        """完全本番設定の統合制覇"""
        settings = Settings(
            environment=Environment.PRODUCTION,
            app_name="Production App",
            debug=False,
            log_level="WARNING",
            api_host="prod.example.com",
            api_port=443,
            cors_origins="https://prod.example.com,https://www.prod.example.com",
            api_keys="prod-key-1,prod-key-2",
            sentry_dsn="https://sentry.io/production",
            cache_ttl_seconds=1800,
            max_workers=8
        )
        
        assert settings.is_production
        assert not settings.debug
        assert len(settings.cors_origins_list) == 2
        assert "https://prod.example.com" in settings.cors_origins_list
        
        issues = settings.validate_config()
        assert len(issues) == 0
        
        db_url = settings.get_database_url()
        assert "postgresql" in db_url
        
        log_config = settings.get_log_config()
        assert log_config["root"]["level"] == "WARNING"
        assert log_config["handlers"]["default"]["formatter"] == "detailed"

    def test_development_to_production_migration_destroyer(self):
        """開発から本番への移行設定制覇"""
        # 開発設定
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        dev_issues = dev_settings.validate_config()
        assert len(dev_issues) == 0
        
        # 本番移行設定（問題あり）
        prod_settings = Settings(
            environment=Environment.PRODUCTION,
            debug=True,  # 移行忘れ
            cors_origins="*"  # 制限忘れ
        )
        prod_issues = prod_settings.validate_config()
        assert len(prod_issues) > 0
        assert any("Debug mode" in issue for issue in prod_issues)
        assert any("CORS should be restricted" in issue for issue in prod_issues)