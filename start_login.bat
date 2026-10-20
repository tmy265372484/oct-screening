@echo off
cd /d "%~dp0"
set OCT_DEPLOYED=0
set OCT_AUTH_MODE=accounts
set OCT_CASE_DB=
set "OCT_DATA_DIR=%~dp0outputs\accounts"
set "OCT_AUTH_DB=%~dp0outputs\auth\accounts.sqlite3"
set "OCT_MODEL_PATH=%~dp0models\torch_resume\best.pt"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port=8502
) else (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port=8502
  ) else (
    py -3.11 -m streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port=8502
  )
)
pause
