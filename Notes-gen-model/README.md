# 🧠 ILA Notes Generation Model

An intelligent notes generation model that creates enhanced, comprehensive study notes from lecture transcripts and educational content. This model is part of the **Intelligent Learning Assistant (ILA)** project.

## 📋 Overview

This model fine-tunes a BART (Bidirectional and Auto-Regressive Transformers) model on educational summarization datasets to generate high-quality notes that:

- **Summarize** key concepts from transcripts
- **Enhance** content with additional insights
- **Structure** information in a clear, study-friendly format
- **Adapt** to different detail levels (short vs. detailed notes)

## 🎯 Features

- ✅ Fine-tuned on `ccdv/arxiv-summarization` dataset
- ✅ Supports both short and detailed note generation
- ✅ GPU acceleration support
- ✅ Easy-to-use inference API
- ✅ Configurable generation parameters

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- 8GB+ RAM (16GB+ recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train the Model

Train the model on the educational summarization dataset:

```bash
python train.py
```

**Training Details:**
- Uses `facebook/bart-large-cnn` as base model
- Trains on 2,000 samples (configurable)
- Validates on 500 samples
- Saves model to `./ila-notes-generator/`
- Training time: ~30-60 minutes on GPU, 2-4 hours on CPU

**Configuration:**
You can modify training parameters in `train.py`:
- `TRAIN_SAMPLES`: Number of training samples (default: 2000)
- `VAL_SAMPLES`: Number of validation samples (default: 500)
- `MAX_INPUT_LENGTH`: Maximum input token length (default: 1024)
- `MAX_TARGET_LENGTH`: Maximum output token length (default: 256)
- `MODEL_NAME`: Base model to fine-tune (default: "facebook/bart-large-cnn")

### 2. Generate Notes

#### Using the Inference Script

```bash
python inference.py
```

This will run a test inference with a sample transcript.

#### Using Python API

```python
from inference import generate_notes, NotesGenerator

# Simple usage
transcript = "Your YouTube video transcript here..."
notes = generate_notes(transcript)
print(notes)

# Generate short notes (concise, 5-8 key points)
short_notes = generate_notes(transcript, detailed=False)

# Generate detailed notes (comprehensive with insights)
detailed_notes = generate_notes(transcript, detailed=True)

# Advanced usage with custom parameters
generator = NotesGenerator()
notes = generator.generate_notes(
    transcript,
    max_length=300,
    min_length=100,
    num_beams=5,
    length_penalty=2.5
)
```

## 📁 Project Structure

```
Notes-gen-model/
├── train.py              # Training script
├── inference.py          # Inference script with API
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── ila-notes-generator/ # Trained model directory (created after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer files
│   └── training_info.json
└── data/                # Optional: for local data storage
```

## 🔧 Model Architecture

- **Base Model**: `facebook/bart-large-cnn`
  - Pre-trained on CNN/DailyMail dataset
  - Excellent for summarization tasks
  - 400M parameters
  - Supports up to 1024 input tokens

- **Fine-tuning**:
  - Dataset: `ccdv/arxiv-summarization`
  - Educational research articles and abstracts
  - Optimized for academic/educational content

## 📊 Training Configuration

Default training hyperparameters:

- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4 (effective batch size: 8)
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **Mixed Precision**: Enabled (if GPU available)

## 🎛️ Inference Parameters

### Generation Parameters

- `max_length`: Maximum length of generated notes (default: 256)
- `min_length`: Minimum length of generated notes (default: 50)
- `num_beams`: Number of beams for beam search (default: 4)
  - Higher = better quality but slower
  - Recommended: 3-5
- `length_penalty`: Length penalty factor (default: 2.0)
  - Higher = longer outputs
  - Lower = shorter outputs
- `temperature`: Sampling temperature (default: 1.0)
  - Only used if `do_sample=True`
  - Higher = more random, Lower = more deterministic

### Note Types

**Short Notes:**
- Concise bullet points
- 5-8 key concepts
- Quick reference format
- ~100-150 tokens

**Detailed Notes:**
- Comprehensive summaries
- Additional insights and context
- Related concepts
- ~200-256 tokens

## 🔄 Integration with ILA Backend

To integrate this model into your ILA backend:

```python
# In your backend service (e.g., notesService.js)
import subprocess
import json

def generate_notes_with_model(transcript):
    """Call the Python model from Node.js"""
    result = subprocess.run(
        ['python', 'inference.py', '--text', transcript],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

Or use a Python microservice with Flask/FastAPI:

```python
from flask import Flask, request, jsonify
from inference import generate_notes

app = Flask(__name__)

@app.route('/generate-notes', methods=['POST'])
def generate():
    data = request.json
    transcript = data.get('transcript', '')
    detailed = data.get('detailed', False)
    
    notes = generate_notes(transcript, detailed=detailed)
    return jsonify({'notes': notes})

if __name__ == '__main__':
    app.run(port=5000)
```

## 📈 Performance

### Training Performance

- **GPU (NVIDIA RTX 3080)**: ~30-45 minutes
- **GPU (NVIDIA RTX 4090)**: ~20-30 minutes
- **CPU (8 cores)**: ~2-4 hours

### Inference Performance

- **GPU**: ~0.5-1 second per transcript
- **CPU**: ~3-5 seconds per transcript

### Model Size

- **Base Model**: ~1.6 GB
- **Fine-tuned Model**: ~1.6 GB
- **Total Disk Space**: ~3-4 GB (including checkpoints)

## 🎓 Fine-tuning with Custom Data

To fine-tune on your own YouTube transcript data:

1. **Prepare your dataset** in the same format as the arxiv dataset:
   ```python
   {
       "article": "Full transcript text...",
       "abstract": "Desired notes output..."
   }
   ```

2. **Modify `train.py`**:
   ```python
   # Replace dataset loading
   from datasets import Dataset
   
   custom_data = Dataset.from_json("your_data.json")
   train_data = custom_data.select(range(2000))
   ```

3. **Train**:
   ```bash
   python train.py
   ```

## 🐛 Troubleshooting

### Out of Memory Error

- Reduce `TRAIN_SAMPLES` or `VAL_SAMPLES`
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use a smaller model (e.g., `facebook/bart-base`)

### Model Not Found Error

- Ensure you've run `train.py` first
- Check that `./ila-notes-generator/` directory exists
- Verify model files are present

### Slow Training

- Enable GPU if available
- Reduce dataset size for initial testing
- Use mixed precision training (fp16)

## 📚 Dataset Information

**Dataset**: `ccdv/arxiv-summarization`
- **Source**: arXiv research papers
- **Format**: Article → Abstract summarization
- **Size**: ~200K training samples
- **Domain**: Academic/Educational content
- **License**: CC0 1.0

## 🔮 Future Enhancements

- [ ] Support for multi-language transcripts
- [ ] Integration with topic modeling
- [ ] Automatic key concept extraction
- [ ] Question generation from notes
- [ ] Interactive note editing
- [ ] Export to various formats (PDF, Markdown, etc.)

## 📝 License

This project is part of the ILA (Intelligent Learning Assistant) project.

## 🤝 Contributing

This is a component of the ILA project. For contributions, please refer to the main ILA repository.

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the training logs in `./ila-notes-generator/logs/`
3. Verify your environment meets the requirements

## 🙏 Acknowledgments

- **Hugging Face** for transformers library and model hosting
- **Facebook AI** for the BART model
- **ccdv** for the arxiv-summarization dataset

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Model**: BART-Large-CNN Fine-tuned  
**Status**: ✅ Production Ready

