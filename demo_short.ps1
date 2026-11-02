$ErrorActionPreference = 'Stop'
chcp 65001 | Out-Null
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    Write-Host "缺少 .venv，请先运行 setup.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "===== 输入设备 =====" -ForegroundColor Cyan
$devices = & $py -m tt_voice.cli devices
$devices | ForEach-Object { Write-Host "  $_" }

$dev = $null
foreach ($pattern in @('\[(\d+)\].*UU远程.*WASAPI', '\[(\d+)\].*UU远程')) {
    $hit = $devices | Select-String -Pattern $pattern | Select-Object -First 1
    if ($hit -and $hit.Line -match '\[(\d+)\]') { $dev = $Matches[1]; break }
}
if (-not $dev) {
    $hit = $devices | Select-String -Pattern '\[(\d+)\].*WASAPI' |
        Where-Object { $_.Line -notmatch 'Realtek|Stereo' } | Select-Object -First 1
    if ($hit -and $hit.Line -match '\[(\d+)\]') { $dev = $Matches[1] }
}
if (-not $dev) { $dev = Read-Host "输入麦克风编号" }
Write-Host "使用设备：$dev" -ForegroundColor Green

Write-Host ""
Write-Host "============== 短视频演示 ==============" -ForegroundColor Cyan
Write-Host " 1) 小裁判 → 左方加一分    （有反馈：播报 + 屏幕 A 1:0 B）"
Write-Host " 2) 小裁判 → 今天天气不错  （无反馈：不播报、比分不变）"
Write-Host " 3) 小裁判 → 撤销上一分    （有反馈：播报 + 屏幕回到 0:0）"
Write-Host " 每次唤醒后 6 秒内说，说完停一下。"
Write-Host "======================================"
Write-Host ""
Start-Sleep -Seconds 2

& $py -m tt_voice.cli mic --no-reask --fresh --say --verbose --device $dev
