# 📓 Colab Notebook - Copy-Paste Ready

Copy each cell into a separate Colab cell and run in order.

---

## Cell 1: Install Dependencies

```python
!pip install -q transformers>=4.35.0 datasets>=2.14.0 torch>=2.0.0 tqdm accelerate sentencepiece protobuf numpy pandas scikit-learn sentence-transformers faiss-cpu peft rouge-score nltk python-dotenv fastapi uvicorn google-generativeai flask flask-cors
print("✅ Dependencies installed")
```

---

## Cell 2: Check GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU! Go to Runtime → Change runtime type → GPU")
```

---

## Cell 3: Create Directories

```python
!mkdir -p src/utils data examples
print("✅ Directories created")
print("\n📁 Now upload your files:")
print("   - train.py, config.py, utils.py, inference.py, enhanced_notes.py")
print("   - src/ folder (all Python files)")
print("   - data/transcripts_sample.jsonl")
```

---

## Cell 4: Upload Files

**Use Colab file browser (📁 icon on left) to upload:**
- All Python files
- `src/` folder
- `data/transcripts_sample.jsonl`

**Or use code:**

```python
from google.colab import files
uploaded = files.upload()  # Select your files
```

---

## Cell 5: Train Model

```python
!python train.py
```

⏱️ **Time:** ~30-60 minutes

---

## Cell 6: Set Up RAG Pipeline

```python
from src.ingestion import TranscriptIngester
from src.indexing import build_index_from_data

print("Processing transcripts...")
ingester = TranscriptIngester(chunk_size=500, chunk_overlap=50)
ingester.process_jsonl('data/transcripts_sample.jsonl', 'data/processed_chunks.json')
print("✅ Transcripts processed")

print("\nBuilding vector index...")
build_index_from_data('data/processed_chunks.json', 'knowledge_index')
print("✅ Vector index built")
```

---

## Cell 7: Set Up Gemini API (Optional)

```python
# Create .env file
with open('.env', 'w') as f:
    f.write("""USE_GEMINI_API=true
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-1.5-pro
TOP_K_RETRIEVAL=6
""")

print("✅ .env file created")
print("⚠️  Edit .env file and add your Gemini API key!")
print("Get it from: https://makersuite.google.com/app/apikey")
```

**Then:**
1. Click on `.env` file in Colab file browser
2. Edit it and replace `YOUR_KEY_HERE` with your actual API key
3. Save

---

## Cell 8: Test Hybrid RAG

```python
from src.rag_inference import RAGInference
import json

# Initialize RAG
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
print(notes.get('summary', 'N/A'))
print(f"\n🧩 Key Concepts: {len(notes.get('key_concepts', []))}")
```

---

## Cell 9: Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

print("Saving to Drive...")
!cp -r ila-notes-generator /content/drive/MyDrive/ILA_Model/
!cp knowledge_index.* /content/drive/MyDrive/ILA_Model/ 2>/dev/null || echo ""
!cp .env /content/drive/MyDrive/ILA_Model/ 2>/dev/null || echo ""

print("✅ Saved to /content/drive/MyDrive/ILA_Model/")
```

---

## Cell 10: Download (Alternative)

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

## Quick Test (After Setup)

```python
# Quick test
from src.rag_inference import RAGInference

rag = RAGInference(
    model_path="./ila-notes-generator",
    index_path="./knowledge_index"
)

notes = rag.generate_notes(
    "Your transcript here...",
    query="Explain this topic"
)

print(notes)
```

---

**That's it!** Follow the cells in order. 🚀

