@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting agents...
start "Reviewer A"  cmd /k "call venv\Scripts\activate.bat && python agents\reviewer_a\main.py"
start "Reviewer B"  cmd /k "call venv\Scripts\activate.bat && python agents\reviewer_b\main.py"
start "Judge"       cmd /k "call venv\Scripts\activate.bat && python agents\judge\main.py"

echo Waiting for agents to initialize...
timeout /t 8 /nobreak > nul

echo Starting orchestrator...
start "Orchestrator" cmd /k "call venv\Scripts\activate.bat && uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000"

echo.
echo All services running!
echo   App  ^-^> http://localhost:8000
echo.
pause