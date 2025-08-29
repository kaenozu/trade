import re
p='pyproject.toml'
s=open(p,'r',encoding='utf-8').read()
s=re.sub(r"(?ms)(\[tool\.ruff\.lint\][\s\S]*?ignore\s*=\s*\[)([\s\S]*?)(\])",
         lambda m: m.group(1)+"\n    \"E501\", # line length handled by formatter\n    \"B008\", # allow FastAPI Depends in params\n"+m.group(3), s)
open(p,'w',encoding='utf-8').write(s)
print('patched ruff ignore list')
