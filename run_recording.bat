@echo off
cd /d "%~dp0"
chcp 65001 >nul
title Voice module real-person recording
echo ==========================================
echo   Real-person acceptance recording
echo ==========================================
echo.
echo Available input devices:
echo.
".venv\Scripts\python.exe" -m tt_voice.cli devices
echo.
echo Look for "UU remote virtual audio device". Common numbers: 12 or 17.
echo.
set /p DEV=Enter device number: 
echo.
echo For each item: press Enter, wait for two beeps, read the sentence, pause.
echo Press Ctrl+C to stop early. Partial data is saved.
echo.
pause
echo.
".venv\Scripts\python.exe" tools\record_dataset.py --out realdata2 --device %DEV% --wake-repeats 5
echo.
echo Finished. Tell Codex: recording done.
pause
