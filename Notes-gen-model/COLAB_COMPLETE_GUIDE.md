# 🚀 Complete Colab Guide - Hybrid RAG Model

Step-by-step guide to run the entire hybrid RAG model in Google Colab with free GPU.

## Step 1: Open Colab & Enable GPU

1. Go to **https://colab.research.google.com/**
2. Click **"New Notebook"**
3. Click **Runtime** → **Change runtime type**
4. Set **Hardware accelerator** to **GPU (T4)**
5. Click **Save**

## Step 2: Install Dependencies

Copy and paste this in a cell:

```python
!pip install -q transformers>=4.35.0 datasets>=2.14.0 torch>=2.0.0 tqdm accelerate sentencepiece protobuf numpy pandas scikit-learn sentence-transformers faiss-cpu peft rouge-score nltk python-dotenv fastapi uvicorn google-generativeai flask flask-cors
```

⏱️ **Time:** ~5-10 minutes

## Step 3: Verify GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU! Go to Runtime → Change runtime type → GPU")
```

## Step 4: Create Directory Structure

```python
!mkdir -p src/utils data examples
print("✅ Directories created")
```

## Step 5: Upload Project Files

**Method 1: Upload via Colab UI (Recommended)**

1. Click the **📁 folder icon** on the left sidebar
2. Click **Upload** button
3. Upload these files one by one:

**Core Files:**
- `train.py`
- `config.py`
- `utils.py`
- `inference.py`
- `enhanced_notes.py`
- `prepare_training_data.py`

**RAG Files (in src/ folder):**
- `src/rag_inference.py`
- `src/indexing.py`
- `src/ingestion.py`
- `src/train_lora.py`
- `src/evaluate.py`
- `src/api_rag.py`
- `src/__init__.py`

**Utils (in src/utils/ folder):**
- `src/utils/prompt_builder.py`
- `src/utils/__init__.py`

**Data:**
- `data/transcripts_sample.jsonl`

**Method 2: Upload as Zip**

1. Zip your project folder locally
2. Upload the zip file
3. Extract it:

```python
from google.colab import files
import zipfile

uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted {filename}")
```

## Step 6: Train the Base Model

```python
!python train.py
```

⏱️ **Time:** ~30-60 minutes on GPU

**What happens:**
- Downloads BART model (~1.6 GB)
- Downloads training dataset
- Trains on 2,000 samples
- Saves to `./ila-notes-generator/`

## Step 7: Set Up RAG Pipeline

```python
# Step 7a: Process transcripts
from src.ingestion import TranscriptIngester

ingester = TranscriptIngester(chunk_size=500, chunk_overlap=50)
ingester.process_jsonl('data/transcripts_sample.jsonl', 'data/processed_chunks.json')
print("✅ Transcripts processed")

# Step 7b: Build vector index
from src.indexing import build_index_from_data

build_index_from_data('data/processed_chunks.json', 'knowledge_index')
print("✅ Vector index built")
```

⏱️ **Time:** ~2-5 minutes

## Step 8: Set Up Gemini API (Optional but Recommended)

```python
# Create .env file
with open('.env', 'w') as f:
    f.write("""USE_GEMINI_API=true
GEMINI_API_KEY=your_google_generativeai_key_here
GEMINI_MODEL=gemini-1.5-pro
TOP_K_RETRIEVAL=6
""")

print("✅ .env file created")
print("⚠️  Don't forget to add your Gemini API key!")
print("Get it from: https://makersuite.google.com/app/apikey")
```

**Then edit the `.env` file** and add your actual API key.

## Step 9: Test Hybrid RAG

```python
from src.rag_inference import RAGInference
import json

# Initialize RAG (auto-detects Gemini from .env)
rag = RAGInference(
    model_path="./ila-notes-generator",
    index_path="./knowledge_index"
)

# Test
transcript = """
Neural networks are computational models inspired by biological neurons.
They consist of layers of interconnected nodes with adjustable weights.
Training involves forward propagation and backpropagation.
"""

print("Generating notes...")
notes = rag.generate_notes(
    transcript,
    query="Explain neural networks for beginners",
    top_k=5
)

print(f"\n✅ Generated using: {notes.get('generation_method', 'unknown')}")
print(f"\n📝 Summary:")
print(notes.get('summary', 'N/A')[:200])
print(f"\n🧩 Key Concepts: {len(notes.get('key_concepts', []))}")
```

## Step 10: Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

print("Saving model and index to Drive...")
!cp -r ila-notes-generator /content/drive/MyDrive/ILA_Model/
!cp knowledge_index.* /content/drive/MyDrive/ILA_Model/ 2>/dev/null || echo "Index files saved"
!cp .env /content/drive/MyDrive/ILA_Model/ 2>/dev/null || echo ""

print("✅ Saved to /content/drive/MyDrive/ILA_Model/")
```

## Step 11: Download (Alternative to Drive)

```python
from google.colab import files
import shutil

# Download model
shutil.make_archive('ila-notes-generator', 'zip', 'ila-notes-generator')
files.download('ila-notes-generator.zip')

# Download index
!zip -r knowledge_index.zip knowledge_index.* 2>/dev/null
files.download('knowledge_index.zip')

print("✅ Downloads started!")
```

---

## Complete Colab Notebook (Copy-Paste Ready)

Here's the complete notebook you can copy-paste:

```python
# ============================================
# CELL 1: Install Dependencies
# ============================================
!pip install -q transformers>=4.35.0 datasets>=2.14.0 torch>=2.0.0 tqdm accelerate sentencepiece protobuf numpy pandas scikit-learn sentence-transformers faiss-cpu peft rouge-score nltk python-dotenv fastapi uvicorn google-generativeai flask flask-cors

# ============================================
# CELL 2: Check GPU
# ============================================
import torch
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# CELL 3: Create Directories
# ============================================
!mkdir -p src/utils data examples
print("✅ Directories created")
print("\n📁 Now upload your files via Colab file browser:")
print("   - train.py, config.py, utils.py, inference.py, enhanced_notes.py")
print("   - src/ folder (all Python files)")
print("   - data/transcripts_sample.jsonl")

# ============================================
# CELL 4: Train Model (Skip if already trained)
# ============================================
# Uncomment to train:
# !python train.py

# ============================================
# CELL 5: Set Up RAG Pipeline
# ============================================
from src.ingestion import TranscriptIngester
from src.indexing import build_index_from_data

print("Processing transcripts...")
ingester = TranscriptIngester()
ingester.process_jsonl('data/transcripts_sample.jsonl', 'data/processed_chunks.json')

print("Building vector index...")
build_index_from_data('data/processed_chunks.json', 'knowledge_index')
print("✅ RAG pipeline ready!")

# ============================================
# CELL 6: Set Up Gemini (Optional)
# ============================================
with open('.env', 'w') as f:
    f.write("""USE_GEMINI_API=true
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-1.5-pro
TOP_K_RETRIEVAL=6
""")
print("✅ .env created - Add your Gemini API key!")

# ============================================
# CELL 7: Test Hybrid RAG
# ============================================
from src.rag_inference import RAGInference
import json

rag = RAGInference(
    model_path="./ila-notes-generator",
    index_path="./knowledge_index"
)

transcript = "Neural networks are computational models..."
notes = rag.generate_notes(transcript, query="Explain neural networks")
print(json.dumps(notes, indent=2)[:500])

# ============================================
# CELL 8: Save to Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')
!cp -r ila-notes-generator /content/drive/MyDrive/ILA_Model/
!cp knowledge_index.* /content/drive/MyDrive/ILA_Model/
print("✅ Saved to Drive!")
```

---

## Quick Reference

### Essential Commands

```python
# Install
!pip install -q transformers datasets torch sentence-transformers faiss-cpu peft google-generativeai

# Train
!python train.py

# Setup RAG
from src.ingestion import TranscriptIngester
from src.indexing import build_index_from_data
ingester = TranscriptIngester()
ingester.process_jsonl('data/transcripts_sample.jsonl', 'data/processed_chunks.json')
build_index_from_data('data/processed_chunks.json', 'knowledge_index')

# Use
from src.rag_inference import RAGInference
rag = RAGInference(model_path="./ila-notes-generator", index_path="./knowledge_index")
notes = rag.generate_notes(transcript)
```

---

## Free Tier Suitability

### ✅ Free Tier is ENOUGH!

**Why:**
- ✅ GPU (T4) is fast enough
- ✅ RAM (12-15 GB) is sufficient  
- ✅ Training completes in ~30-60 minutes
- ✅ Can save to Drive for persistence

**Limitations:**
- ⚠️ 12-hour session limit (save frequently)
- ⚠️ May disconnect after inactivity
- ⚠️ ~12 hours GPU/day (enough for training)

**Tips:**
- Save to Drive after each major step
- Download model immediately after training
- Use checkpoints during training

---

## Troubleshooting

### "Module not found"
→ Make sure you uploaded all files in correct folders

### "Index not found"
→ Run Step 7 to build the index

### "Model not found"
→ Run Step 6 to train the model

### "Gemini API error"
→ Check your API key in `.env` file

### Session disconnected
→ Load from Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/ILA_Model/ila-notes-generator ./
!cp /content/drive/MyDrive/ILA_Model/knowledge_index.* ./
```

---

## Complete Workflow Summary

1. **Open Colab** → Enable GPU
2. **Install dependencies** (Step 2)
3. **Upload files** (Step 5)
4. **Train model** (Step 6) - ~30-60 min
5. **Set up RAG** (Step 7) - ~5 min
6. **Set up Gemini** (Step 8) - Optional
7. **Test** (Step 9)
8. **Save to Drive** (Step 10)

**Total Time:** ~1-2 hours (mostly training)

---

**You're all set!** 🚀

