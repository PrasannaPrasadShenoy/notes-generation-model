@echo off
echo ========================================
echo Local Setup - Hybrid RAG Model
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

echo [4/6] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo Continue anyway? (Y/N)
    pause
)

echo [5/6] Checking for .env file...
if not exist .env (
    echo Creating .env file template...
    (
        echo USE_GEMINI_API=true
        echo GEMINI_API_KEY=your_google_generativeai_key_here
        echo GEMINI_MODEL=gemini-1.5-pro
        echo TOP_K_RETRIEVAL=6
    ) > .env
    echo [INFO] .env file created. Please edit it with your API key.
) else (
    echo .env file already exists.
)

echo [6/6] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARNING] PyTorch verification failed
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit .env file with your Gemini API key (optional)
echo   2. Train model: python train.py
echo   3. Setup RAG: python run_rag_pipeline.py
echo   4. Test: python test_rag.py
echo   5. Run API: python -m uvicorn src.api_rag:app --reload
echo.
pause

