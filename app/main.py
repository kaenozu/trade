import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .services import data as data_service
from .services.features import build_feature_frame
from .services.model import predict_future, train_or_load_model
from .services.signals import generate_trade_plan
from .services.tickers import list_jp_tickers

app = FastAPI(title="JP Stocks ML Forecaster", version="0.1.0")


class PredictionRequest(BaseModel):
    ticker: str
    horizon_days: int = 10
    lookback_days: int = 400


class PredictionPoint(BaseModel):
    date: str
    expected_return: float
    expected_price: float


class TradePlan(BaseModel):
    buy_date: str | None
    sell_date: str | None
    confidence: float
    rationale: str


class PredictionResponse(BaseModel):
    ticker: str
    horizon_days: int
    trade_plan: TradePlan
    predictions: list[PredictionPoint]


class Quote(BaseModel):
    ticker: str
    price: float
    asof: str


class QuoteItem(BaseModel):
    ticker: str
    price: float | None = None
    asof: str | None = None
    error: str | None = None


class BulkQuotesResponse(BaseModel):
    quotes: list[QuoteItem]


@app.get("/", response_class=HTMLResponse)
def index():
    return """
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
                 html+=`<tr><td>${r.ticker}</td><td>${r.name}</td><td>${r.sector}</td><td><button onclick="run('${r.ticker}','${r.name.replace(/'/g, "\\\\'")}')">予測</button></td></tr>`
               }
               html+='</tbody></table>';
               document.getElementById('list').innerHTML=html;
               // 価格列を挿入しつつ非同期で取得
               const rows = Array.from(document.querySelectorAll('#list tbody tr'));
               const pairs = rows.map((tr,i)=>{ const cell=tr.insertCell(3); cell.textContent='…'; return {ticker: data[i].ticker, el: cell}; });
               // 一括取得で現在値を埋める
               try{
                 const tickers = data.map(r=>r.ticker).join(',');
                 const res2 = await fetch('/quotes?tickers='+encodeURIComponent(tickers));
                 if(res2.ok){
                   const payload = await res2.json();
                   const map = new Map(payload.quotes.map(q=>[q.ticker, q]));
                   for(const it of pairs){
                     const q = map.get(it.ticker);
                     if(q && q.price!=null) it.el.textContent = Number(q.price).toFixed(2);
                     else it.el.textContent = '—';
                   }
                 } else {
                   for(const it of pairs){ it.el.textContent = '—'; }
                 }
               }catch(_){
                 for(const it of pairs){ it.el.textContent = '—'; }
               }
            }
            async function run(ticker,name){
               const horizon=parseInt(document.getElementById('horizon').value)||10;
               const lookback=parseInt(document.getElementById('lookback').value)||400;
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
               document.getElementById('plan').textContent=`買: ${tp.buy_date || '-'}
売: ${tp.sell_date || '-'}
確信度: ${tp.confidence.toFixed(2)}
理由: ${tp.rationale}`
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


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        df = data_service.fetch_ohlcv(req.ticker, period_days=req.lookback_days + 5)
    except ValueError as e:`n            raise HTTPException(status_code=400, detail=str(e)) from e from e

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
def tickers(q: str | None = None):
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
        except ValueError as e:`n            raise HTTPException(status_code=400, detail=str(e)) from e from e
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data")
        last_idx: pd.Timestamp = df.index.max()  # type: ignore
        last_close = float(df.loc[last_idx, "Close"])  # type: ignore
        return Quote(ticker=ticker, price=last_close, asof=str(last_idx.date()))  # type: ignore


@app.get("/quotes", response_model=BulkQuotesResponse)
def quotes(tickers: str):
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers is required")
    raw = [t.strip() for t in tickers.split(",") if t.strip()]
    # de-dup and cap to 300
    seen = []
    for t in raw:
        if t not in seen:
            seen.append(t)
        if len(seen) >= 300:
            break
    out: list[QuoteItem] = []
    for t in seen:
        try:
            price, asof = data_service.fetch_last_close_direct(t)
            out.append(QuoteItem(ticker=t, price=price, asof=asof))
        except Exception as e:
            # Fallback to OHLCV
            try:
                df = data_service.fetch_ohlcv(t, period_days=60)
                if len(df) > 0:
                    last_idx: pd.Timestamp = df.index.max()  # type: ignore
                    last_close = float(df.loc[last_idx, "Close"])  # type: ignore
                    out.append(QuoteItem(ticker=t, price=last_close, asof=str(last_idx.date())))
                else:
                    out.append(QuoteItem(ticker=t, error=str(e)))
            except Exception as e2:
                out.append(QuoteItem(ticker=t, error=str(e2)))
    return BulkQuotesResponse(quotes=out)

