param(
  [int]$Port = 8000,
  [switch]$NoReload
)

$ErrorActionPreference = 'Stop'

# Stop existing uvicorn on the same app if running
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'python(.exe)?\s+-m\s+uvicorn\s+app.main:app' } |
  ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }

$reloadFlag = if ($NoReload) { '' } else { ' --reload' }
$args = "-m uvicorn app.main:app --host 0.0.0.0 --port $Port$reloadFlag"

Write-Host "Starting: python $args"
python $args

