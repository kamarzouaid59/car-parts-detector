@echo off
setlocal

set DAMAGE_INPUT_MODE=url
set DAMAGE_SERVICE_URL=http://127.0.0.1:8001
set PREDICT_WITH_DAMAGE=true

cd /d "%~dp0"
uv run ex2.py --server
