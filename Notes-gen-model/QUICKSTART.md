# 🚀 Quick Start Guide

Get started with the ILA Notes Generation Model in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will install PyTorch, Transformers, and other required libraries. If you have a GPU, PyTorch will automatically detect it.

## Step 2: Train the Model

```bash
python train.py
```

**What happens:**
- Downloads the `ccdv/arxiv-summarization` dataset
- Loads the BART-Large-CNN model
- Fine-tunes on 2,000 training samples
- Saves model to `./ila-notes-generator/`

**Time:** 
- GPU: ~30-60 minutes
- CPU: ~2-4 hours

**Tip:** You can reduce `TRAIN_SAMPLES` in `config.py` for faster training during testing.

## Step 3: Generate Notes

### Option A: Test with Sample Data

```bash
python inference.py
```

This will run inference on a sample transcript and show you:
- Short notes
- Detailed notes  
- Enhanced notes (with metadata)

### Option B: Use in Your Code

```python
from inference import generate_notes

# Your YouTube transcript
transcript = "Your video transcript here..."

# Generate short notes
short_notes = generate_notes(transcript, detailed=False)

# Generate detailed notes
detailed_notes = generate_notes(transcript, detailed=True)

# Generate enhanced notes (recommended - includes metadata)
enhanced_notes = generate_notes(transcript, enhanced=True)

print(enhanced_notes)
```

### Option C: Advanced Usage

```python
from inference import NotesGenerator

generator = NotesGenerator()

# Custom parameters
notes = generator.generate_notes(
    transcript,
    max_length=300,
    min_length=100,
    num_beams=5,
    length_penalty=2.5
)
```

## Step 4: Integrate with ILA Backend

### Python Integration

```python
# In your notesService.py or similar
from inference import generate_notes

def generate_notes_for_video(transcript):
    return generate_notes(transcript, enhanced=True)
```

### Node.js Integration (via subprocess)

```javascript
// In your Node.js backend
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

async function generateNotes(transcript) {
  const { stdout } = await execPromise(
    `python inference.py --text "${transcript}"`
  );
  return JSON.parse(stdout);
}
```

### REST API (Flask Example)

Create `api_server.py`:

```python
from flask import Flask, request, jsonify
from inference import generate_notes

app = Flask(__name__)

@app.route('/generate-notes', methods=['POST'])
def generate():
    data = request.json
    transcript = data.get('transcript', '')
    detailed = data.get('detailed', False)
    enhanced = data.get('enhanced', True)
    
    notes = generate_notes(transcript, detailed=detailed, enhanced=enhanced)
    return jsonify({'notes': notes})

if __name__ == '__main__':
    app.run(port=5000)
```

Then call from your ILA backend:

```javascript
// In your Node.js backend
const axios = require('axios');

async function generateNotes(transcript) {
  const response = await axios.post('http://localhost:5000/generate-notes', {
    transcript: transcript,
    enhanced: true
  });
  return response.data.notes;
}
```

## 📊 Understanding the Output

### Short Notes
- Concise bullet points
- 5-8 key concepts
- Quick reference format
- ~100-150 words

### Detailed Notes
- Comprehensive summaries
- Additional context
- Structured format
- ~200-256 words

### Enhanced Notes (Recommended)
- Everything in detailed notes
- **Plus:**
  - Reading time estimate
  - Key topics/phrases
  - Formatted for easy reading

## ⚙️ Configuration

Edit `config.py` to customize:

- **Training**: `TRAIN_SAMPLES`, `NUM_EPOCHS`, `LEARNING_RATE`
- **Generation**: `MAX_INPUT_LENGTH`, `MAX_TARGET_LENGTH`
- **Note Types**: Adjust beam search, length penalty, etc.

## 🐛 Troubleshooting

### "Model not found" error
→ Run `python train.py` first

### Out of memory during training
→ Reduce `TRAIN_SAMPLES` or `TRAIN_BATCH_SIZE` in `config.py`

### Slow inference
→ Use GPU if available, or reduce `num_beams` parameter

## 📚 Next Steps

1. **Fine-tune on your data**: Replace dataset in `train.py` with your YouTube transcripts
2. **Adjust parameters**: Experiment with generation parameters for better results
3. **Integrate**: Connect to your ILA backend
4. **Monitor**: Track note quality and user feedback

## 💡 Tips

- **For best results**: Use `enhanced=True` - it adds useful metadata
- **For speed**: Use `detailed=False` for quick summaries
- **For quality**: Increase `num_beams` to 5-6 (slower but better)
- **For customization**: Use `NotesGenerator` class for full control

---

**Ready to go!** 🎉

For more details, see the full [README.md](README.md).

