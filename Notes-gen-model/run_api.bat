@echo off
echo ========================================
echo Starting Hybrid RAG API Server
echo ========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

echo Starting FastAPI server...
echo.
echo Server will be available at:
echo   - API: http://127.0.0.1:8000
echo   - Docs: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m uvicorn src.api_rag:app --reload --host 127.0.0.1 --port 8000

pause

