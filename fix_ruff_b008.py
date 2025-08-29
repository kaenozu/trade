import re
p='pyproject.toml'
s=open(p,'r',encoding='utf-8').read()
s=re.sub(r"(\[tool\.ruff\.lint\][\s\S]*?ignore\s*=\s*)\[[^\]]*\]", r"\1[\n    \"E501\", # line length handled by formatter\n    \"B008\", # allow FastAPI Depends in params\n]", s)
open(p,'w',encoding='utf-8').write(s)
print('pyproject updated')
