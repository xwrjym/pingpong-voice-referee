<#
一键安装脚本（在目标电脑上运行，把整个文件夹拷过去后执行）：

    powershell -ExecutionPolicy Bypass -File setup.ps1

可选参数：
    -SkipInstall      跳过创建虚拟环境和安装依赖
    -SkipModel        跳过模型下载（包内已带模型时会自动跳过）
    -Mirror <url>     pip 镜像，默认阿里云
#>
param(
    [switch]$SkipInstall,
    [switch]$SkipModel,
    [string]$Mirror = "https://mirrors.aliyun.com/pypi/simple/"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$modelDir = Join-Path $PSScriptRoot "models\vosk-model-small-cn-0.22"

function Step([string]$title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

Step "1/6 检查 Python"
$pythonCmd = "python"
try {
    $versionText = & $pythonCmd -V 2>&1 | Out-String
    Write-Host "  找到 $($versionText.Trim())"
} catch {
    Write-Host "  找不到 python 命令。请先安装 Python 3.10 或更高版本（勾选 Add to PATH）后重跑本脚本。" -ForegroundColor Red
    exit 1
}
if ($versionText -notmatch "3\.(1[0-9]|[2-9][0-9])") {
    Write-Host "  Python 版本过低，本项目用到 3.10+ 的语法。" -ForegroundColor Red
    exit 1
}

if (-not $SkipInstall) {
    Step "2/6 创建虚拟环境并安装依赖"
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "  python -m venv .venv"
        & $pythonCmd -m venv (Join-Path $PSScriptRoot ".venv")
    }
    Write-Host "  pip install -r requirements.txt -i $Mirror"
    & $python -m pip install --disable-pip-version-check -q -i $Mirror -r requirements.txt
} else {
    Step "2/6 跳过依赖安装"
}
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "  虚拟环境不存在：$python" -ForegroundColor Red
    exit 1
}

Step "3/6 检查识别模型"
if (Test-Path -LiteralPath $modelDir) {
    Write-Host "  已就绪：$modelDir"
} elseif ($SkipModel) {
    Write-Host "  未找到模型，且指定了 -SkipModel。可改用零模型后端：--engine dtw" -ForegroundColor Yellow
} else {
    Write-Host "  未找到模型，开始下载（约 42MB，国内建议挂代理）……"
    try {
        & $python tools\download_model.py
    } catch {
        Write-Host "  下载失败。可改用零模型后端，或从另一台电脑直接拷 models\ 目录过来。" -ForegroundColor Yellow
    }
}

Step "4/6 列出输入设备（记下麦克风编号，后面用 --device 指定）"
& $python -m tt_voice.cli devices

Step "5/6 自检"
& $python -m tt_voice.cli selftest

Step "6/6 纯文本跑一遍状态机（不依赖麦克风，验证业务逻辑）"
& $python -m tt_voice.cli text 小裁判 我加一分 对方加一分 刚才不算

Write-Host ""
Write-Host "安装完成。接下来：" -ForegroundColor Green
Write-Host "  1) 实时试：   .venv\Scripts\python.exe -m tt_voice.cli mic --say --verbose"
Write-Host "     （打不开麦克风就加 --device <编号>，编号取第 4 步的输出）"
Write-Host "  2) 真人验收： .venv\Scripts\python.exe tools\record_dataset.py --out realdata"
Write-Host "                .venv\Scripts\python.exe -m tt_voice.cli eval realdata"
Write-Host "  3) 想用 DTW 后端（零模型）：tools\enroll.py --mic --repeats 3，然后 --engine dtw"
