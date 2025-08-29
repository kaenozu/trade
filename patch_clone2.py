# patch model.py in branch #33
p1='app/services/model.py'
s=open(p1,'r',encoding='utf-8').read()
if 'import re' not in s.splitlines()[0:20]:
    s=s.replace('from dataclasses import dataclass','from dataclasses import dataclass\nimport re')
s=s.replace('\nndef _validate_ticker','\ndef _validate_ticker')
# normalize base line
s = '\n'.join(['    base = ticker.replace("/", "_").replace("\\\\", "_")' if l.strip().startswith('base = ticker.replace(') else l for l in s.splitlines()]) + '\n'
open(p1,'w',encoding='utf-8').write(s)

# patch data.py indentation
p2='app/services/data.py'
ls=open(p2,'r',encoding='utf-8').read().splitlines()
for i,l in enumerate(ls):
    if l.strip()=="ticker = _validate_ticker(ticker)" and i>110 and i<130:
        ls[i]='    ticker = _validate_ticker(ticker)'
for i in range(len(ls)):
    if ls[i].strip().startswith('if os.path.exists(cache_file)'):
        base=i
        ls[base]='    if os.path.exists(cache_file) and (now - os.path.getmtime(cache_file) <= ttl_seconds):'
        ls[base+1]='        try:'
        ls[base+2]='            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)'
        ls[base+3]='        except Exception:'
        ls[base+4]='            df = pd.DataFrame()'
        ls[base+5]='    else:'
        ls[base+6]='        df = pd.DataFrame()'
        # remove duplicate next line if exists
        if base+7 < len(ls) and ls[base+7].strip()=='df = pd.DataFrame()':
            ls[base+7]=''
        break
open(p2,'w',encoding='utf-8').write('\n'.join(ls)+'\n')
print('patched clone2 files')
