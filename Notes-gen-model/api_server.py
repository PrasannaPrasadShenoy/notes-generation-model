"""
REST API Server for ILA Notes Generation Model
Provides HTTP endpoints for easy integration with ILA backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import NotesGenerator, generate_notes
from enhanced_notes import EnhancedNotesGenerator
from utils import validate_transcript, estimate_reading_time, extract_key_phrases
import os
import sys
import logging
from datetime import datetime

# Add src to path for RAG imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.rag_inference import RAGInference
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG modules not available. Install dependencies: pip install sentence-transformers faiss-cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global model instances (loaded once at startup)
generator = None
enhanced_generator = None
rag_inference = None

def init_model():
    """Initialize the model on server startup"""
    global generator, enhanced_generator, rag_inference
    try:
        logger.info("Loading notes generation model...")
        generator = NotesGenerator()
        enhanced_generator = EnhancedNotesGenerator()
        logger.info("✅ Model loaded successfully!")
        
        # Try to initialize RAG (optional)
        if RAG_AVAILABLE:
            try:
                model_path = os.getenv("MODEL_PATH", "./ila-notes-generator")
                index_path = os.getenv("INDEX_PATH", "./knowledge_index")
                
                if os.path.exists(model_path) and os.path.exists(f"{index_path}.index"):
                    rag_inference = RAGInference(
                        model_path=model_path,
                        index_path=index_path,
                        use_gemini=None  # Auto-detect
                    )
                    if rag_inference.use_gemini and rag_inference.gemini_client:
                        logger.info(f"✅ Hybrid RAG loaded: Gemini API + Local Retrieval")
                    elif rag_inference.model:
                        logger.info("✅ RAG loaded: Local Model + Local Retrieval")
                else:
                    logger.info("ℹ️  RAG not initialized (model or index not found)")
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
        
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
        
        # Initialize metadata
        metadata = {}
        
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
            # Use enhanced generator for v2 features
            enhanced_result = enhanced_generator.generate_enhanced_notes_v2(transcript)
            notes = enhanced_result['notes']
            # Include additional metadata from enhanced generator
            if include_metadata:
                metadata = {
                    'key_concepts': enhanced_result['enhancements'].get('key_concepts', []),
                    'study_tips': enhanced_result['enhancements'].get('study_tips', []),
                    'related_topics': enhanced_result['enhancements'].get('related_topics', []),
                    'learning_objectives': enhanced_result['enhancements'].get('learning_objectives', []),
                    'reading_time_minutes': enhanced_result['metadata']['reading_time_minutes'],
                    'word_count': enhanced_result['metadata']['word_count'],
                    'key_topics': enhanced_result['metadata']['key_topics']
                }
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid note type: {note_type}. Must be "short", "detailed", or "enhanced"'
            }), 400
        
        # Extract metadata if not already set (for short/detailed notes)
        if include_metadata and not metadata:
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

@app.route('/api/generate-enhanced-v2', methods=['POST'])
def generate_enhanced_v2():
    """
    Generate enhanced notes v2 with all features (key concepts, study tips, etc.)
    
    Request body:
    {
        "transcript": "Your transcript text here...",
        "include_concepts": true,
        "include_tips": true,
        "include_topics": true,
        "include_objectives": true
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
        
        # Generate enhanced notes v2
        result = enhanced_generator.generate_enhanced_notes_v2(
            transcript,
            include_concepts=data.get('include_concepts', True),
            include_tips=data.get('include_tips', True),
            include_topics=data.get('include_topics', True),
            include_objectives=data.get('include_objectives', True)
        )
        
        return jsonify({
            'success': True,
            'notes': result['notes'],
            'base_notes': result['base_notes'],
            'enhancements': result['enhancements'],
            'metadata': result['metadata']
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating enhanced notes v2: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-hybrid-rag', methods=['POST'])
def generate_hybrid_rag():
    """
    Generate notes using Hybrid RAG (Local Retrieval + Gemini/Local Generation)
    
    Request body:
    {
        "transcript": "Your transcript text here...",
        "query": "Explain this topic",
        "top_k": 6,
        "force_local": false,
        "force_gemini": false
    }
    """
    if not RAG_AVAILABLE or not rag_inference:
        return jsonify({
            'success': False,
            'error': 'RAG system not available. Make sure model and index are set up. See RAG_SETUP_GUIDE.md'
        }), 503
    
    try:
        data = request.get_json()
        transcript = data.get('transcript', '').strip()
        query = data.get('query')
        top_k = data.get('top_k')
        force_local = data.get('force_local', False)
        force_gemini = data.get('force_gemini', False)
        
        is_valid, error = validate_transcript(transcript)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Generate with hybrid RAG
        notes = rag_inference.generate_notes(
            transcript=transcript,
            query=query,
            top_k=top_k,
            force_local=force_local,
            force_gemini=force_gemini
        )
        
        return jsonify({
            'success': True,
            'notes': notes,
            'generation_method': notes.get('generation_method', 'unknown'),
            'used_gemini': notes.get('generation_method') == 'gemini',
            'retrieval_count': notes.get('retrieval_count', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in hybrid RAG generation: {e}", exc_info=True)
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

