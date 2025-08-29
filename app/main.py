from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

# Optional observability (non-fatal if missing)
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
except Exception:  # pragma: no cover
    Instrumentator = None  # type: ignore

try:
    import sentry_sdk  # type: ignore
    from sentry_sdk.integrations.fastapi import FastApiIntegration  # type: ignore
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore
    FastApiIntegration = None  # type: ignore

from .services import data as data_service
from .services.features import build_feature_frame
from .services.model import train_or_load_model, predict_future
from .services.signals import generate_trade_plan
from .services.tickers import list_jp_tickers

app = FastAPI(title="JP Stocks ML Forecaster", version="0.1.0")

# Sentry (enabled when SENTRY_DSN is provided)
_sentry_dsn = os.environ.get("SENTRY_DSN")
if _sentry_dsn and sentry_sdk and FastApiIntegration:  # pragma: no cover
    sentry_sdk.init(
        dsn=_sentry_dsn,
        integrations=[FastApiIntegration()],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
        environment=os.environ.get("SENTRY_ENV", "production"),
    )

# Prometheus metrics (enabled by default; disable via METRICS_ENABLED=0)
if os.environ.get("METRICS_ENABLED", "1") not in ("0", "false", "False") and Instrumentator:  # pragma: no cover
    try:
        Instrumentator().instrument(app).expose(app, include_in_schema=False)
    except Exception:
        pass


class PredictionRequest(BaseModel):
    ticker: str
    horizon_days: int = 10
    lookback_days: int = 400


class PredictionPoint(BaseModel):
    date: str
    expected_return: float
    expected_price: float


class TradePlan(BaseModel):
    buy_date: Optional[str]
    sell_date: Optional[str]
    confidence: float
    rationale: str


class PredictionResponse(BaseModel):
    ticker: str
    horizon_days: int
    trade_plan: TradePlan
    predictions: List[PredictionPoint]


class Quote(BaseModel):
    ticker: str
    price: float
    asof: str


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        """
        <!doctype html>
        <html>
        <head>
          <meta charset='utf-8' />
          <meta name='viewport' content='width=device-width, initial-scale=1' />
          <title>JP Stocks ML Forecaster</title>
          <style>
            body{font-family:system-ui,Arial;margin:24px;}
            input,button{font-size:16px;padding:8px;margin-right:8px}
            .row{margin:8px 0}
            table{border-collapse:collapse;margin-top:12px}
            th,td{border:1px solid #ddd;padding:6px}
          </style>
        </head>
        <body>
          <h2>日本株 予測・売買タイミング（デモ）</h2>
          <div class='row'>
            <input id='q' placeholder='銘柄名/コード/セクターで検索' style='width:320px' />
            <button onclick='loadTickers()'>検索</button>
          </div>
          <div class='row'>
            <label>予測日数</label>
            <input id='horizon' type='number' value='10' min='1' max='30' />
            <label>学習過去日数</label>
            <input id='lookback' type='number' value='400' min='200' max='1200' />
          </div>
          <div id='list'></div>
          <hr/>
          <h3 id='selTitle'>選択銘柄: -</h3>
          <pre id='plan'></pre>
          <div id='preds'></div>
          <script>
            async function loadTickers(){
               const q=document.getElementById('q').value.trim();
               const res=await fetch('/tickers'+(q?`?q=${encodeURIComponent(q)}`:''));
               const data=await res.json();
               let html='<table><thead><tr><th>コード</th><th>名称</th><th>セクター</th><th>現在値</th><th></th></tr></thead><tbody>';
               for(const r of data){
                 html+=`<tr><td>${r.ticker}</td><td>${r.name}</td><td>${r.sector}</td><td><button onclick="run('${r.ticker}','${r.name.replace(/'/g, "\'")}')">予測</button></td></tr>`
               }
               html+='</tbody></table>';
               document.getElementById('list').innerHTML=html;
               // 価格列を挿入しつつ非同期で取得
               const rows = Array.from(document.querySelectorAll('#list tbody tr'));
               const pairs = rows.map((tr,i)=>{ const cell=tr.insertCell(3); cell.textContent='…'; return {ticker: data[i].ticker, el: cell}; });
               const concurrency=8; let k=0;
               async function worker(){
                 while(k < pairs.length){
                   const it = pairs[k++];
                   try{
                     const res = await fetch('/quote?ticker='+encodeURIComponent(it.ticker));
                     if(!res.ok) throw new Error(String(res.status));
                     const q = await res.json();
                     it.el.textContent = Number(q.price).toFixed(2);
                   }catch(_){
                     it.el.textContent = '—';
                   }
                 }
               }
               await Promise.all(Array(concurrency).fill(0).map(()=>worker()));
            }
            async function run(ticker,name){
               const horizon=parseInt(document.getElementById('horizon').value,10);
               const lookback=parseInt(document.getElementById('lookback').value,10);
               document.getElementById('selTitle').textContent=`選択銘柄: ${ticker} ${name||''}`
               document.getElementById('plan').textContent='Running...';
               const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker,horizon_days:horizon,lookback_days:lookback})});
               if(!res.ok){
                 try{
                   const errText = await res.text();
                   let msg = 'Error: '+res.status;
                   try{
                     const errJson = JSON.parse(errText);
                     if(errJson && (errJson.detail || errJson.message)){
                       msg += ' - ' + (errJson.detail || errJson.message);
                     }
                   }catch(_){
                     if(errText) msg += ' - '+errText;
                   }
                   document.getElementById('plan').textContent = msg;
                 }catch(_){
                   document.getElementById('plan').textContent='Error: '+res.status;
                 }
                 return;
               }
               const data=await res.json();
               const tp=data.trade_plan;
               document.getElementById('plan').textContent=`買: ${tp.buy_date || '-'}\n売: ${tp.sell_date || '-'}\n確信度: ${tp.confidence.toFixed(2)}\n理由: ${tp.rationale}`
               let html='<table><thead><tr><th>日付</th><th>期待リターン</th><th>予測価格</th></tr></thead><tbody>';
               for(const p of data.predictions){
                 html += `<tr><td>${p.date}</td><td>${(p.expected_return*100).toFixed(2)}%</td><td>${p.expected_price.toFixed(2)}</td></tr>`
               }
               html+='</tbody></table>'
               document.getElementById('preds').innerHTML=html;
            }
            loadTickers();
          </script>
          <p style='margin-top:12px;color:#666'>免責事項: 本ツールは教育目的のデモです。投資判断はご自身の責任で行ってください。過去実績は将来の結果を保証しません。</p>
        </body>
        </html>
        """
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        df = data_service.fetch_ohlcv(req.ticker, period_days=req.lookback_days + 5)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df) < 200:
        raise HTTPException(status_code=400, detail="Not enough data to train")

    feat = build_feature_frame(df)
    model, meta = train_or_load_model(req.ticker, feat)
    pred_df = predict_future(df, feat, model, horizon_days=req.horizon_days)
    trade = generate_trade_plan(pred_df)

    preds = []
    for idx, row in pred_df.iterrows():
        preds.append(
            PredictionPoint(
                date=str(pd.to_datetime(idx).date()),
                expected_return=float(row["expected_return"]),
                expected_price=float(row["expected_price"]),
            )
        )

    return PredictionResponse(
        ticker=req.ticker,
        horizon_days=req.horizon_days,
        trade_plan=TradePlan(**trade),
        predictions=preds,
    )
@app.get("/tickers")
def tickers(q: Optional[str] = None):
    return list_jp_tickers(query=q)


@app.get("/quote", response_model=Quote)
def quote(ticker: str):
    # Basic input sanitation (also validated in data layer)
    if not ticker or len(ticker) > 15:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    # Prefer direct lightweight quote for robustness
    try:
        price, asof = data_service.fetch_last_close_direct(ticker)
        return Quote(ticker=ticker, price=price, asof=asof)
    except Exception:
        # Fallback to OHLCV fetch
        try:
            df = data_service.fetch_ohlcv(ticker, period_days=90)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data")
        last_idx = df.index.max()
        last_close = float(df.loc[last_idx, "Close"])
        return Quote(ticker=ticker, price=last_close, asof=str(pd.to_datetime(last_idx).date()))
