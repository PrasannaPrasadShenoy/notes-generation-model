# 💻 Local Setup Guide - Hybrid RAG Model

Complete guide to run the hybrid RAG model on your local Windows machine.

## Prerequisites

1. **Python 3.8+** installed
   - Check: `python --version`
   - Download: https://www.python.org/downloads/

2. **pip** (usually comes with Python)
   - Check: `pip --version`

3. **Git** (optional, for cloning)
   - Download: https://git-scm.com/download/win

4. **GPU (Optional but Recommended)**
   - NVIDIA GPU with CUDA support
   - Check: `nvidia-smi` in Command Prompt
   - If no GPU, CPU will work but slower

---

## Step 1: Navigate to Project Directory

Open **PowerShell** or **Command Prompt** and navigate to your project:

```powershell
cd "C:\Users\aweso\Downloads\ML model\notes-generation-model\Notes-gen-model"
```

---

## Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again
```

**You should see `(venv)` in your prompt.**

---

## Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

⏱️ **Time:** ~10-15 minutes

**If you get errors:**
- **"Microsoft Visual C++ 14.0 is required"** → Install Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- **"torch installation failed"** → Install PyTorch separately:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  (Replace `cu118` with your CUDA version or `cpu` if no GPU)

---

## Step 4: Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
python -c "import faiss; print('FAISS: OK')"
```

---

## Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```powershell
# Create .env file
New-Item -Path .env -ItemType File -Force
```

**Edit `.env` file** (use Notepad or any text editor):

```env
# Gemini API (Optional - for hybrid RAG)
USE_GEMINI_API=true
GEMINI_API_KEY=your_google_generativeai_key_here
GEMINI_MODEL=gemini-1.5-pro
TOP_K_RETRIEVAL=6

# Model Paths (Optional - defaults work)
MODEL_PATH=./ila-notes-generator
INDEX_PATH=./knowledge_index
```

**Get Gemini API Key:**
1. Go to: https://makersuite.google.com/app/apikey
2. Create API key
3. Copy and paste into `.env` file

---

## Step 6: Train the Model

### Option A: Quick Training (Recommended for first time)

```powershell
python train.py
```

⏱️ **Time:** 
- **With GPU:** ~30-60 minutes
- **Without GPU:** ~2-4 hours

**What happens:**
- Downloads BART model (~1.6 GB)
- Downloads training dataset
- Trains on 2,000 samples
- Saves to `./ila-notes-generator/`

### Option B: Custom Training (with your own data)

```powershell
# First, prepare your data
python prepare_training_data.py

# Then train
python train_custom.py
```

---

## Step 7: Set Up RAG Pipeline

```powershell
# Process transcripts
python -c "from src.ingestion import TranscriptIngester; i = TranscriptIngester(); i.process_jsonl('data/transcripts_sample.jsonl', 'data/processed_chunks.json')"

# Build vector index
python -c "from src.indexing import build_index_from_data; build_index_from_data('data/processed_chunks.json', 'knowledge_index')"
```

**Or use the automated script:**

```powershell
python run_rag_pipeline.py
```

⏱️ **Time:** ~2-5 minutes

---

## Step 8: Test the Model

### Test 1: Basic Inference

```powershell
python inference.py
```

### Test 2: Enhanced Notes

```powershell
python -c "from enhanced_notes import EnhancedNotesGenerator; gen = EnhancedNotesGenerator(); notes = gen.generate('Your transcript here...'); print(notes)"
```

### Test 3: Hybrid RAG

```powershell
python example_hybrid_rag.py
```

**Or create a test script:**

```python
# test_rag.py
from src.rag_inference import RAGInference

rag = RAGInference(
    model_path="./ila-notes-generator",
    index_path="./knowledge_index"
)

transcript = """
Neural networks are computational models inspired by biological neurons.
They consist of layers of interconnected nodes with adjustable weights.
Training involves forward propagation and backpropagation.
"""

notes = rag.generate_notes(
    transcript,
    query="Explain neural networks for beginners",
    top_k=5
)

print(f"Generated using: {notes.get('generation_method', 'unknown')}")
print(f"\nSummary: {notes.get('summary', 'N/A')}")
print(f"\nKey Concepts: {notes.get('key_concepts', [])}")
```

Run it:
```powershell
python test_rag.py
```

---

## Step 9: Run API Server

### Option A: Flask API (Original)

```powershell
python api_server.py
```

Server runs at: **http://127.0.0.1:5000**

### Option B: FastAPI (RAG-specific)

```powershell
python -m uvicorn src.api_rag:app --reload --host 127.0.0.1 --port 8000
```

Server runs at: **http://127.0.0.1:8000**

**Test the API:**

```powershell
# PowerShell
$body = @{
    transcript = "Neural networks are computational models..."
    query = "Explain neural networks"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/generate" -Method POST -Body $body -ContentType "application/json"
```

**Or use curl (if installed):**

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"transcript\": \"Your transcript...\", \"query\": \"explain this\"}"
```

---

## Step 10: Access API Documentation

If using FastAPI, visit:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## Complete Workflow Summary

```powershell
# 1. Navigate to project
cd "C:\Users\aweso\Downloads\ML model\notes-generation-model\Notes-gen-model"

# 2. Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (edit with your API key)
# (Create .env manually or use: New-Item -Path .env -ItemType File)

# 5. Train model
python train.py

# 6. Set up RAG
python run_rag_pipeline.py

# 7. Test
python example_hybrid_rag.py

# 8. Run API server
python -m uvicorn src.api_rag:app --reload
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size in `config.py`
- Use CPU instead (slower but works)
- Close other GPU-intensive applications

### Issue: "FileNotFoundError: knowledge_index.index"

**Solution:**
```powershell
# Rebuild the index
python run_rag_pipeline.py
```

### Issue: "Gemini API error"

**Solution:**
- Check `.env` file has correct API key
- Verify API key is valid at: https://makersuite.google.com/app/apikey
- Check internet connection

### Issue: "Port already in use"

**Solution:**
```powershell
# Use a different port
python -m uvicorn src.api_rag:app --reload --port 8001
```

### Issue: "Execution Policy Error" (PowerShell)

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Performance Tips

### With GPU:
- Training: ~30-60 minutes
- Inference: ~1-3 seconds per note
- RAG retrieval: ~0.1-0.5 seconds

### Without GPU (CPU only):
- Training: ~2-4 hours
- Inference: ~5-15 seconds per note
- RAG retrieval: ~0.5-2 seconds

**Recommendation:** Use GPU if available for much faster training and inference.

---

## File Structure After Setup

```
Notes-gen-model/
├── venv/                    # Virtual environment
├── .env                     # Environment variables (create this)
├── ila-notes-generator/     # Trained model (created after training)
├── knowledge_index.index    # FAISS index (created after RAG setup)
├── knowledge_index_metadata.json
├── data/
│   ├── transcripts_sample.jsonl
│   └── processed_chunks.json  # Created after ingestion
├── src/
│   ├── rag_inference.py
│   ├── indexing.py
│   └── ...
└── ...
```

---

## Quick Commands Reference

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Train model
python train.py

# Set up RAG
python run_rag_pipeline.py

# Test RAG
python example_hybrid_rag.py

# Run Flask API
python api_server.py

# Run FastAPI
python -m uvicorn src.api_rag:app --reload

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Next Steps

1. ✅ Train the model
2. ✅ Set up RAG pipeline
3. ✅ Test locally
4. ✅ Run API server
5. ✅ Integrate with your frontend

**You're all set!** 🚀

