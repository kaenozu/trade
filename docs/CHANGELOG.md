# 変更履歴

最終更新: 2025-08-29

## 2025-08-29

- 機能追加: 現在値APIとUI
  - `GET /quote?ticker=XXXX` 単銘柄の現在値取得を追加。
  - `GET /quotes?tickers=a,b,c` 一括取得で一覧のN+1を解消。
  - トップページ一覧に「現在値」列を追加し、読み込み時に `/quotes` で一括反映。

- データ取得の堅牢化（`app/services/data.py`）
  - ティッカー簡易バリデーションを追加（`^[A-Za-z0-9._-]{1,15}）。
  - `requests.Session` に UA/ヘッダを設定する `_make_session()` を追加（Proxy環境にも配慮）。
  - YahooチャートAPI直叩きの `fetch_last_close_direct()` を実装（軽量な現在値取得）。
  - `fetch_ohlcv()` にセッションを渡し yfinance 取得の安定性を向上、キャッシュ/整形を維持。

- API/モデル（`app/main.py`）
  - モデル: `Quote`, `QuoteItem`, `BulkQuotesResponse` を追加。
  - ルート: `/quote`, `/quotes` を追加。
  - 既存のマージコンフリクトマーカーを解消。

- 依存関係
  - `requirements.txt` に `requests>=2.31.0` を追加。

- コード品質の改善
  - 未使用のインポートを削除 (ruff)
  - `__all__` に型注釈を追加 (mypy)
  - `mypy` の型エラーを修正

- ログ/その他
  - `.gitignore` に `uvicorn.out.log` / `uvicorn.err.log` が既に含まれていることを確認。
  - ただし sparse-checkout 下でインデックスからの除外は未実施（必要なら `git rm --cached --sparse` 等で対応）。

- ドキュメント
  - 旧 `docs/WORKLOG.md` は指示に従って削除。

## 実行/確認

- 起動: `uvicorn app.main:app --host 127.0.0.1 --port 8000`
- ルート: `http://127.0.0.1:8000/`
- API: `/quote?ticker=7203.T`, `/quotes?tickers=7203.T,9984.T,6758.T`
