param(
  [string]$Repo = "kaenozu/trade",
  [string]$PlanPath = "docs/issues_plan.md"
)

if (!(Test-Path $PlanPath)) { Write-Error "Plan file not found: $PlanPath"; exit 1 }

$content = Get-Content -Path $PlanPath -Raw
$blocks = ($content -split "`n`n-") | Where-Object { $_ -match 'タイトル:' }

function New-Issue($title, $body, $labels) {
  if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh issue create --repo $Repo --title $title --body $body --label $labels
  } elseif ($env:GH_TOKEN) {
    $api = "https://api.github.com/repos/$Repo/issues"
    $json = @{ title = $title; body = $body; labels = ($labels -split ',\s*') } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri $api -Headers @{ Authorization = "token $($env:GH_TOKEN)" } -Body $json
  } else {
    Write-Host "[DRY-RUN]" $title
  }
}

foreach ($b in $blocks) {
  $title = [regex]::Match($b, 'タイトル:\s*(.+)') .Groups[1].Value.Trim()
  $labels = [regex]::Matches($b, 'ラベル:\s*(.+)') | Select-Object -Last 1 | ForEach-Object { $_.Groups[1].Value.Trim() }
  $bodyLines = @()
  foreach ($line in ($b -split "`n")) { if ($line -match '^\s*-\s*内容:' -or $line -match '^\s*-\s*受入条件:') { $bodyLines += $line.Trim() } }
  $body = ($bodyLines -join "`n")
  if ($title) { New-Issue -title $title -body $body -labels $labels }
}

Write-Host "Done."
