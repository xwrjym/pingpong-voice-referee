@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 语音模块 真人录音

echo ==========================================
echo   语音模块 真人验收录音
echo ==========================================
echo.
echo [1] 当前可用的输入设备：
echo.
".venv\Scripts\python.exe" -m tt_voice.cli devices
echo.
echo     请认准名字里带 "UU远程虚拟音频设备" 的那一个（常见是 12 或 17）。
echo.
set /p DEV=请输入麦克风编号，然后按回车：
echo.
echo 每一条的操作：按回车 - 听到两短音 - 念屏幕上的字 - 停一下等它自动结束。
echo 中途想停可以按 Ctrl+C，已录部分会保存。
echo.
pause
echo.
".venv\Scripts\python.exe" tools\record_dataset.py --out realdata2 --device %DEV% --wake-repeats 5
echo.
echo 录音结束。把这个窗口的内容截图发给 Codex，或直接说"录完了"。
pause
