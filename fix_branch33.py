import re
# fix model.py
p1='app/services/model.py'
s=open(p1,'r',encoding='utf-8').read()
if 'import re' not in s.split('\n')[0:20]:
    s=s.replace('from dataclasses import dataclass','from dataclasses import dataclass\nimport re')
s=s.replace('\nndef _validate_ticker','\ndef _validate_ticker')
s=s.replace('replace(\\"\\\\\\", \"_\")','replace("\\\\", "_")')
open(p1,'w',encoding='utf-8').write(s)

# fix data.py indentation and duplicate line
p2='app/services/data.py'
d=open(p2,'r',encoding='utf-8').read().splitlines()
for i,l in enumerate(d):
    if l.strip()=="ticker = _validate_ticker(ticker)" and i>100 and i<140:
        d[i]='    '+d[i]  # ensure correct indent inside function header? We'll remove instead
for i,l in enumerate(d):
    if l.strip()=="ticker = _validate_ticker(ticker)" and i>120:
        d[i] = ''
for i,l in enumerate(d):
    if l.strip().startswith('if os.path.exists(cache_file)'):
        base=i
        break
else:
    base=None
if base is not None:
    # rebuild the next 7 lines to the correct block
    d[base] = '    if os.path.exists(cache_file) and (now - os.path.getmtime(cache_file) <= ttl_seconds):'
    d[base+1] = '        try:'
    d[base+2] = '            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)'
    d[base+3] = '        except Exception:'
    d[base+4] = '            df = pd.DataFrame()'
    d[base+5] = '    else:'
    d[base+6] = '        df = pd.DataFrame()'
open(p2,'w',encoding='utf-8').write('\n'.join(d)+'\n')
print('patched files')
