# JP Stocks ML Forecaster

日本株の終値データから機械学習で短期リターンを予測し、予測区間内での「買い・売り」タイミングを提示する FastAPI アプリです。

## セットアップ

```
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 起動

```
uvicorn app.main:app --reload --port 8000
```

ブラウザで `http://127.0.0.1:8000/` を開き、銘柄（例: `7203.T`）を入力して実行します。

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

