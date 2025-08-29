"""Frontend/UI endpoints."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    """Serve the main application HTML interface."""
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
  <div class='row' style='color:#555'>一覧から銘柄を選んで「予測」を押してください（予測日数=10日、学習期間=400日）。</div>
  <div id='list'></div>
  <hr/>
  <h3 id='selTitle'>選択銘柄: -</h3>
  <pre id='plan'></pre>
  <div id='preds'></div>
  <script>
    async function loadTickers(){
       const res=await fetch('/tickers');
       const data=await res.json();
       let html='<table><thead><tr><th>コード</th><th>名称</th><th>セクター</th><th>現在値</th><th></th></tr></thead><tbody>';
       for(const r of data){
         html+=`<tr><td>${r.ticker}</td><td>${r.name}</td><td>${r.sector}</td><td><button onclick="run('${r.ticker}','${r.name.replace(/'/g, "\\'")}')">予測</button></td></tr>`
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
       const horizon=10; // default
       const lookback=400; // default
       document.getElementById('selTitle').textContent=`選択銘柄: ${ticker} ${name||''}`
       document.getElementById('plan').textContent='Running...';
       const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker,horizon_days:horizon,lookback_days:lookback})});
       if(!res.ok){
         try{
           const errText = await res.text();
           let msg = 'Error: '+res.status;
           try{
             const errJson = JSON.parse(errText);
             if(errJson.detail) msg += ' - '+errJson.detail;
           }catch(_){}
           document.getElementById('plan').textContent = msg;
         }catch(_){
           document.getElementById('plan').textContent = 'Error: '+res.status;
         }
         return;
       }
       const payload = await res.json();
       const plan = payload.trade_plan;
       let planText = `Buy: ${plan.buy_date||'N/A'}, Sell: ${plan.sell_date||'N/A'}\\nConfidence: ${(plan.confidence*100).toFixed(1)}%\\n${plan.rationale}`;
       document.getElementById('plan').textContent = planText;
       let html='<table><thead><tr><th>Date</th><th>Expected Return</th><th>Expected Price</th></tr></thead><tbody>';
       for(const p of payload.predictions){
         html+=`<tr><td>${p.date}</td><td>${(p.expected_return*100).toFixed(2)}%</td><td>${p.expected_price.toFixed(2)}</td></tr>`;
       }
       html+='</tbody></table>';
       document.getElementById('preds').innerHTML=html;
    }
    loadTickers();
  </script>
  <p style='margin-top:12px;color:#666'>免責事項: 本ツールは教育目的のデモです。投資判断はご自身の責任で行ってください。過去実績は将来の結果を保証しません。</p>
</body>
</html>
"""