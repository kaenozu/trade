# 日本株ML予測・売買タイミング Webアプリ 仕様書

## 目的
- 日本株（例: `7203.T`）の株価データから機械学習モデルで短期（1〜30営業日）のリターンを予測し、
  予測区間内での「いつ買って・いつ売るか」を提示する。

## 非機能要件
- 予測精度: 時系列CVにより汎化性能を推定。特徴量・アルゴリズムは軽量で堅牢なもの（HGBRegressor）を採用。
- パフォーマンス: 特徴量は計算量の小さいテクニカル系中心。モデルは軽量GBDT。キャッシュ（モデル永続化）で再学習を削減。
- 信頼性: 例外処理・入力検証。サービス層を分離しテスト容易性を確保。
- テスト: Pytest によるユニットテスト（特徴量、モデル、シグナル、API）。外部API(yfinance) はモック。

## 構成
- `FastAPI` バックエンド + シンプルな組み込みフロント（HTML）。
- サービス層
  - `data`: yfinanceでOHLCV取得（本番）。テストは合成データ使用。
  - `features`: テクニカル指標ベースの特徴量生成（RSI, MACD, 移動平均、ボラ等）＋翌日リターンを教師信号。
  - `model`: HistGradientBoostingRegressor による回帰。時系列CVでR2を推定。`models/` に保存・再利用。
  - `signals`: 予測系列から累積収益最大となる買い・売りの区間を算出。

## API
- `GET /` : 簡易フロント。銘柄・予測日数・学習期間の入力UI。
- `POST /predict`
  - 入力: `{ ticker: string, horizon_days: int(1-30), lookback_days: int(>=200) }`
  - 出力: `{ ticker, horizon_days, trade_plan, predictions[] }`
    - `trade_plan`: `{ buy_date: str|null, sell_date: str|null, confidence: float(0-1), rationale: str }`
    - `predictions[]`: `[{ date: 'YYYY-MM-DD', expected_return: float, expected_price: float }]`

## アルゴリズム概要
- 特徴量: 1/5/10日リターン、対数リターンの分散(5/20日)、移動平均(SMA 5/20/50)、RSI(14)、MACD一式、出来高Zスコア、HLレンジ。
- 予測: 翌日リターンの回帰。直近特徴から逐次的に将来日を予測し、予測価格パスを生成。
- 売買決定: 予測リターン系列でlog(1+r)の累積最大区間を探索→買い日・売り日を提示。信頼度は平均絶対リターンと日数からSigmoidで近似。

## 入力検証 / エラー
- `ticker` の必須・簡易妥当性チェック。
- 学習に必要な最小データ件数（>=200）。
- データ取得失敗や空データは400応答。

## セキュリティ
- 依存の固定化（`requirements.txt`）。
- 入力値の上限（例: horizon<=30, lookback<=1200）。

## 依存関係
- `fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`, `scikit-learn`, `ta`, `yfinance`, `pytest`, `httpx`。

## 運用
- 起動: `uvicorn app.main:app --reload`。
- モデルキャッシュ: `models/<TICKER>.joblib`（特徴量一致時のみ再利用）。

## テスト方針
- ユニット: 特徴量生成の形状・NaN無・十分な行数。
- モデル: 合成データで学習・将来予測が返ること。
- シグナル: 一貫した最適区間選択。
- API: `fetch_ohlcv` をモックしてE2E形状検証。

## 注意・免責
- 本アプリは教育目的のサンプル。市場実運用の前に十分な検証・リスク管理が必要。将来の成果を保証しない。
