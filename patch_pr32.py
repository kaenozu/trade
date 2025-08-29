# patch data.py in feature/bulk-quotes-v2 to harden URL and cache path
p='app/services/data.py'
s=open(p,'r',encoding='utf-8').read()
# import quote
if 'from urllib.parse import quote' not in s:
    s=s.replace('import json','import json\nfrom urllib.parse import quote')
# harden cache path: ensure realpath under cache dir
s=s.replace('return os.path.join(CACHE_DIR, f"yf_{safe}_{period_days}d.csv")',
            'p = os.path.join(CACHE_DIR, f"yf_{safe}_{period_days}d.csv")\n    real = os.path.realpath(p)\n    root = os.path.realpath(CACHE_DIR)\n    if not real.startswith(root + os.sep):\n        raise ValueError("Unsafe cache path")\n    return real')
# quote ticker in URL build
s=s.replace('url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"',
            'safe_ticker = quote(ticker, safe="A-Za-z0-9._-")\n    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{safe_ticker}?interval=1d&range=5d"')
open(p,'w',encoding='utf-8').write(s)
print('patched data.py for #32')
