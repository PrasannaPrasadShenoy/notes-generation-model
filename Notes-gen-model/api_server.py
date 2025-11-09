"""
REST API Server for ILA Notes Generation Model
Provides HTTP endpoints for easy integration with ILA backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import NotesGenerator, generate_notes
from utils import validate_transcript, estimate_reading_time, extract_key_phrases
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global model instance (loaded once at startup)
generator = None

def init_model():
    """Initialize the model on server startup"""
    global generator
    try:
        logger.info("Loading notes generation model...")
        generator = NotesGenerator()
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/generate-notes', methods=['POST'])
def generate_notes_endpoint():
    """
    Generate notes from transcript
    
    Request body:
    {
        "transcript": "Your transcript text here...",
        "type": "short" | "detailed" | "enhanced" (default: "enhanced"),
        "format": "plain" | "bullet" | "numbered" | "markdown" (default: "markdown"),
        "include_metadata": true | false (default: true)
    }
    
    Response:
    {
        "success": true,
        "notes": "Generated notes...",
        "metadata": {
            "reading_time_minutes": 2.5,
            "word_count": 500,
            "key_topics": ["topic1", "topic2", ...]
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        transcript = data.get('transcript', '').strip()
        note_type = data.get('type', 'enhanced').lower()
        format_type = data.get('format', 'markdown').lower()
        include_metadata = data.get('include_metadata', True)
        
        # Validate transcript
        is_valid, error = validate_transcript(transcript)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Generate notes based on type
        if note_type == 'short':
            notes = generator.generate_short_notes(transcript, format_output=(format_type != 'plain'))
        elif note_type == 'detailed':
            notes = generator.generate_detailed_notes(
                transcript, 
                format_output=(format_type != 'plain'),
                include_metadata=include_metadata
            )
        elif note_type == 'enhanced':
            notes = generator.generate_enhanced_notes(transcript)
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid note type: {note_type}. Must be "short", "detailed", or "enhanced"'
            }), 400
        
        # Extract metadata
        metadata = {}
        if include_metadata:
            reading_time = estimate_reading_time(notes)
            key_topics = extract_key_phrases(transcript, max_phrases=10)
            metadata = {
                'reading_time_minutes': reading_time['reading_time_minutes'],
                'word_count': reading_time['word_count'],
                'key_topics': key_topics[:10]
            }
        
        logger.info(f"Generated {note_type} notes for transcript ({len(transcript)} chars)")
        
        return jsonify({
            'success': True,
            'notes': notes,
            'metadata': metadata,
            'type': note_type,
            'format': format_type
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating notes: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-notes/batch', methods=['POST'])
def generate_notes_batch():
    """
    Generate notes for multiple transcripts in batch
    
    Request body:
    {
        "transcripts": [
            {"id": "video1", "transcript": "..."},
            {"id": "video2", "transcript": "..."}
        ],
        "type": "short" | "detailed" | "enhanced",
        "format": "plain" | "bullet" | "numbered" | "markdown"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transcripts' not in data:
            return jsonify({
                'success': False,
                'error': 'No transcripts array provided'
            }), 400
        
        transcripts = data.get('transcripts', [])
        note_type = data.get('type', 'enhanced').lower()
        format_type = data.get('format', 'markdown').lower()
        
        results = []
        
        for item in transcripts:
            transcript_id = item.get('id', 'unknown')
            transcript = item.get('transcript', '').strip()
            
            try:
                is_valid, error = validate_transcript(transcript)
                if not is_valid:
                    results.append({
                        'id': transcript_id,
                        'success': False,
                        'error': error
                    })
                    continue
                
                # Generate notes
                if note_type == 'short':
                    notes = generator.generate_short_notes(transcript, format_output=(format_type != 'plain'))
                elif note_type == 'detailed':
                    notes = generator.generate_detailed_notes(transcript, format_output=(format_type != 'plain'))
                else:
                    notes = generator.generate_enhanced_notes(transcript)
                
                reading_time = estimate_reading_time(notes)
                key_topics = extract_key_phrases(transcript, max_phrases=5)
                
                results.append({
                    'id': transcript_id,
                    'success': True,
                    'notes': notes,
                    'metadata': {
                        'reading_time_minutes': reading_time['reading_time_minutes'],
                        'word_count': reading_time['word_count'],
                        'key_topics': key_topics[:5]
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing transcript {transcript_id}: {e}")
                results.append({
                    'id': transcript_id,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(transcripts),
            'successful': sum(1 for r in results if r.get('success', False))
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-transcript', methods=['POST'])
def analyze_transcript():
    """
    Analyze transcript without generating notes (metadata only)
    
    Request body:
    {
        "transcript": "Your transcript text here..."
    }
    """
    try:
        data = request.get_json()
        transcript = data.get('transcript', '').strip()
        
        is_valid, error = validate_transcript(transcript)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Extract metadata
        word_count = len(transcript.split())
        char_count = len(transcript)
        key_topics = extract_key_phrases(transcript, max_phrases=15)
        estimated_reading_time = estimate_reading_time(transcript)
        
        return jsonify({
            'success': True,
            'analysis': {
                'word_count': word_count,
                'character_count': char_count,
                'estimated_reading_time_minutes': estimated_reading_time['reading_time_minutes'],
                'key_topics': key_topics,
                'topic_count': len(key_topics),
                'is_suitable_for_notes': word_count >= 100
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing transcript: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if generator is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        info_path = os.path.join(generator.model_dir, 'training_info.json')
        training_info = {}
        
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                training_info = json.load(f)
        
        return jsonify({
            'success': True,
            'model_loaded': True,
            'model_dir': generator.model_dir,
            'device': generator.device,
            'training_info': training_info
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize model
    if not init_model():
        logger.error("Failed to initialize model. Exiting...")
        exit(1)
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"🚀 Starting ILA Notes Generation API Server on {host}:{port}")
    logger.info("📚 Available endpoints:")
    logger.info("   POST /api/generate-notes - Generate notes from transcript")
    logger.info("   POST /api/generate-notes/batch - Batch note generation")
    logger.info("   POST /api/analyze-transcript - Analyze transcript metadata")
    logger.info("   GET  /api/model-info - Get model information")
    logger.info("   GET  /health - Health check")
    
    app.run(host=host, port=port, debug=False)

