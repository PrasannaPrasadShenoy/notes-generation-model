# 📡 API Documentation

Complete API documentation for the ILA Notes Generation REST API Server.

## Base URL

```
http://localhost:5000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 2. Generate Notes

**POST** `/api/generate-notes`

Generate notes from a transcript.

**Request Body:**
```json
{
  "transcript": "Your transcript text here...",
  "type": "enhanced",  // "short" | "detailed" | "enhanced"
  "format": "markdown",  // "plain" | "bullet" | "numbered" | "markdown"
  "include_metadata": true
}
```

**Response:**
```json
{
  "success": true,
  "notes": "Generated notes...",
  "metadata": {
    "reading_time_minutes": 2.5,
    "word_count": 500,
    "key_topics": ["topic1", "topic2"],
    "key_concepts": [...],  // Only for enhanced type
    "study_tips": [...],    // Only for enhanced type
    "related_topics": [...], // Only for enhanced type
    "learning_objectives": [...] // Only for enhanced type
  },
  "type": "enhanced",
  "format": "markdown"
}
```

**Note Types:**
- `short`: Concise bullet points (5-8 key concepts)
- `detailed`: Comprehensive summaries with context
- `enhanced`: Full enhanced notes with all features (recommended)

---

### 3. Generate Enhanced Notes V2

**POST** `/api/generate-enhanced-v2`

Generate enhanced notes with all features (key concepts, study tips, etc.).

**Request Body:**
```json
{
  "transcript": "Your transcript text here...",
  "include_concepts": true,
  "include_tips": true,
  "include_topics": true,
  "include_objectives": true
}
```

**Response:**
```json
{
  "success": true,
  "notes": "Full formatted notes with all enhancements...",
  "base_notes": "Base notes without enhancements...",
  "enhancements": {
    "key_concepts": [
      {
        "term": "Machine Learning",
        "explanation": "A subset of AI that enables computers to learn..."
      }
    ],
    "study_tips": [
      "📚 This is a comprehensive topic - break it into smaller sections...",
      "⏰ Review these notes within 24 hours for better retention"
    ],
    "related_topics": ["Deep Learning", "Neural Networks"],
    "learning_objectives": [
      "Understand the basics of machine learning",
      "Learn about different types of ML algorithms"
    ]
  },
  "metadata": {
    "reading_time_minutes": 3.2,
    "word_count": 640,
    "key_topics": ["machine learning", "artificial intelligence", ...]
  }
}
```

---

### 4. Batch Note Generation

**POST** `/api/generate-notes/batch`

Generate notes for multiple transcripts at once.

**Request Body:**
```json
{
  "transcripts": [
    {
      "id": "video1",
      "transcript": "First transcript..."
    },
    {
      "id": "video2",
      "transcript": "Second transcript..."
    }
  ],
  "type": "enhanced",
  "format": "markdown"
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "video1",
      "success": true,
      "notes": "Generated notes...",
      "metadata": {...}
    },
    {
      "id": "video2",
      "success": true,
      "notes": "Generated notes...",
      "metadata": {...}
    }
  ],
  "total": 2,
  "successful": 2
}
```

---

### 5. Analyze Transcript

**POST** `/api/analyze-transcript`

Analyze a transcript without generating notes (metadata only).

**Request Body:**
```json
{
  "transcript": "Your transcript text here..."
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "word_count": 1250,
    "character_count": 7500,
    "estimated_reading_time_minutes": 6.25,
    "key_topics": ["topic1", "topic2", ...],
    "topic_count": 15,
    "is_suitable_for_notes": true
  }
}
```

---

### 6. Model Information

**GET** `/api/model-info`

Get information about the loaded model.

**Response:**
```json
{
  "success": true,
  "model_loaded": true,
  "model_dir": "./ila-notes-generator",
  "device": "cuda",
  "training_info": {
    "model_name": "facebook/bart-large-cnn",
    "dataset": "ccdv/arxiv-summarization",
    "train_samples": 2000,
    "epochs": 3,
    ...
  }
}
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

---

## Usage Examples

### cURL

```bash
# Generate enhanced notes
curl -X POST http://localhost:5000/api/generate-notes \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Your transcript here...",
    "type": "enhanced",
    "include_metadata": true
  }'
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function generateNotes(transcript) {
  const response = await axios.post('http://localhost:5000/api/generate-notes', {
    transcript: transcript,
    type: 'enhanced',
    include_metadata: true
  });
  
  return response.data;
}
```

### Python

```python
import requests

def generate_notes(transcript):
    response = requests.post('http://localhost:5000/api/generate-notes', json={
        'transcript': transcript,
        'type': 'enhanced',
        'include_metadata': True
    })
    return response.json()
```

---

## Integration with ILA Backend

See `integration_example.js` for a complete Node.js integration example.

### Quick Integration

```javascript
// In your notesService.js
const axios = require('axios');

async function generateNotesForVideo(videoId, transcript) {
  try {
    const response = await axios.post('http://localhost:5000/api/generate-enhanced-v2', {
      transcript: transcript,
      include_concepts: true,
      include_tips: true,
      include_topics: true,
      include_objectives: true
    });
    
    if (response.data.success) {
      return {
        videoId: videoId,
        notes: response.data.notes,
        enhancements: response.data.enhancements,
        metadata: response.data.metadata
      };
    }
  } catch (error) {
    console.error('Error generating notes:', error);
    throw error;
  }
}
```

---

## Running the Server

### Development

```bash
python api_server.py
```

### Production (with Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Environment Variables

- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)

```bash
PORT=8080 HOST=0.0.0.0 python api_server.py
```

---

## Rate Limiting

Currently, there's no rate limiting implemented. For production use, consider adding:

- Flask-Limiter for rate limiting
- Authentication/API keys
- Request queuing for high traffic

---

## CORS

CORS is enabled by default for all origins. For production, configure CORS to allow only your frontend domain:

```python
CORS(app, origins=["https://your-frontend-domain.com"])
```

