# ⚡ Local Quick Start - 5 Minutes

Ultra-quick guide to run the hybrid RAG model locally on Windows.

## Step 1: Setup

```powershell
# Navigate to project
cd "C:\Users\aweso\Downloads\ML model\notes-generation-model\Notes-gen-model"

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Create .env File

Create `.env` file in project root:

```env
USE_GEMINI_API=true
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-pro
TOP_K_RETRIEVAL=6
```

## Step 3: Train Model

```powershell
python train.py
```

⏱️ ~30-60 min (GPU) or ~2-4 hours (CPU)

## Step 4: Setup RAG

```powershell
python run_rag_pipeline.py
```

## Step 5: Test

```powershell
python example_hybrid_rag.py
```

## Step 6: Run API

```powershell
# FastAPI
python -m uvicorn src.api_rag:app --reload

# Or Flask
python api_server.py
```

**Access:** http://127.0.0.1:8000

---

**Done!** 🎉

For detailed instructions, see `LOCAL_SETUP_GUIDE.md`

