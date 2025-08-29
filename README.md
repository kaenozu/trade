# JP Stocks ML Forecaster

[![CI](https://github.com/kaenozu/trade/actions/workflows/ci.yml/badge.svg)](https://github.com/kaenozu/trade/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaenozu/trade/branch/main/graph/badge.svg)](https://codecov.io/gh/kaenozu/trade)

日本株の終値データから機械学習で短期リターンを予測し、予測区間内での「買い・売り」タイミングを提示する FastAPI アプリです。

## セットアップ

```
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 起動

- 直接起動:

```
uvicorn app.main:app --reload --port 8000
```

- Windows のスクリプトで起動（ホットリロード有効）:

```
powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 -Port 8000
```

ブラウザで `http://127.0.0.1:8000/` を開き、銘柄（例: `7203.T`）を入力して実行します。

ヘルスチェック: `GET /healthz` は `{ "status": "ok" }` を返します。
バージョン: `GET /version` はアプリ名・バージョン・`GIT_SHA` を返します。

## 運用/監視（オプション）

- メトリクス: `/metrics` にPrometheusメトリクスを公開（デフォルト有効）。無効化は `METRICS_ENABLED=0`。
- エラートラッキング: Sentryを有効化する場合は `SENTRY_DSN` を環境変数で設定。
  - 例: `SENTRY_TRACES_SAMPLE_RATE=0.1`（APM任意）、`SENTRY_PROFILES_SAMPLE_RATE=0.1`、`SENTRY_ENV=prod`。
- CORS: `CORS_ORIGINS`（カンマ区切り）で許可オリジンを設定。デフォルトは `*`。

環境変数の例は `.env.example` を参照してください。

### キャッシュ設定

- `CACHE_DIR`: OHLCV取得のCSVキャッシュ格納先（既定: 作業ディレクトリ配下の`cache/`）
- `ALLOW_SYNTHETIC_DATA=1`: ネットワーク不可や取得失敗時にデモ用の合成データを生成（既定: 0=無効）

## リリース

- タグ付け（例）: `git tag v0.1.0 && git push origin v0.1.0`
- GitHub ActionsがDockerをビルドしGHCRへpush、GitHub Releaseを作成します。
  - イメージ: `ghcr.io/<owner>/<repo>:latest` および `:v0.1.0`

## テスト

```
pytest -q
```

外部API呼び出しはモックしているため、ネットワーク不要で実行可能です。

## 構成

- `app/main.py`: FastAPI エンドポイントと簡易フロント
- `app/services/data.py`: データ取得（yfinance）
- `app/services/features.py`: 特徴量生成
- `app/services/model.py`: 学習・予測・モデル保存
- `app/services/signals.py`: 売買タイミング生成
- `tests/`: ユニットテスト
- `docs/spec.md`: 仕様書

## 免責事項
本ツールは教育目的のデモです。投資判断はご自身の責任で行ってください。過去実績は将来の結果を保証しません。

