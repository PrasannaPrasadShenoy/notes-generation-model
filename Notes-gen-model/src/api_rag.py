"""
Enhanced FastAPI Server with RAG Support
Provides REST API endpoints for RAG-based note generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_inference import RAGInference
from src.evaluate import NoteEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ILA Insight Generator API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_inference = None
evaluator = None


class GenerateRequest(BaseModel):
    transcript: str
    query: Optional[str] = None
    top_k: Optional[int] = None
    use_structured: bool = True
    use_gemini: Optional[bool] = None  # None = auto-detect from config
    force_local: bool = False  # Force local model even if Gemini available


class EvaluateRequest(BaseModel):
    generated: Dict
    reference: Optional[str] = None
    transcript: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global rag_inference, evaluator
    
    model_path = os.getenv("MODEL_PATH", "./ila-notes-generator")
    index_path = os.getenv("INDEX_PATH", "./knowledge_index")
    
    try:
        logger.info("Loading Hybrid RAG inference system...")
        rag_inference = RAGInference(
            model_path=model_path if os.path.exists(model_path) else None,
            index_path=index_path if os.path.exists(f"{index_path}.index") else None,
            use_gemini=None,  # Auto-detect from config/env
            gemini_api_key=None,  # Auto-detect from config/env
            gemini_model=None  # Auto-detect from config/env
        )
        
        if rag_inference.use_gemini and rag_inference.gemini_client:
            logger.info(f"✅ Hybrid RAG loaded: Gemini API ({rag_inference.gemini_model}) + Local Retrieval")
        elif rag_inference.model:
            logger.info("✅ RAG loaded: Local Model + Local Retrieval")
        else:
            logger.warning("⚠️  RAG loaded but no generation backend available")
    except Exception as e:
        logger.warning(f"Could not load RAG system: {e}. Some endpoints may not work.")
        rag_inference = None
    
    evaluator = NoteEvaluator()
    logger.info("✅ Evaluator loaded")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_loaded": rag_inference is not None,
        "evaluator_loaded": evaluator is not None
    }


@app.post("/api/generate-rag")
async def generate_rag_notes(request: GenerateRequest):
    """
    Generate notes using RAG pipeline
    
    Args:
        request: GenerateRequest with transcript and options
    
    Returns:
        Generated notes
    """
    if not rag_inference:
        raise HTTPException(
            status_code=503,
            detail="RAG inference system not loaded. Check MODEL_PATH and INDEX_PATH."
        )
    
    try:
        # Determine if we should use Gemini
        use_gemini = request.use_gemini
        if use_gemini is None:
            # Auto-detect from config
            use_gemini = rag_inference.use_gemini and rag_inference.gemini_client is not None
        
        notes = rag_inference.generate_notes(
            transcript=request.transcript,
            query=request.query,
            top_k=request.top_k,
            use_structured=request.use_structured,
            force_gemini=use_gemini and not request.force_local,
            force_local=request.force_local
        )
        
        return {
            "success": True,
            "notes": notes,
            "generation_method": notes.get('generation_method', 'unknown'),
            "used_gemini": notes.get('generation_method') == 'gemini'
        }
    except Exception as e:
        logger.error(f"Error generating notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
async def evaluate_notes(request: EvaluateRequest):
    """
    Evaluate generated notes
    
    Args:
        request: EvaluateRequest with generated notes and optional reference
    
    Returns:
        Evaluation metrics
    """
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not loaded")
    
    try:
        metrics = evaluator.evaluate_notes(
            generated=request.generated,
            reference=request.reference,
            transcript=request.transcript
        )
        
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error evaluating notes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
async def model_info():
    """Get information about loaded models"""
    info = {
        "rag_loaded": rag_inference is not None,
        "evaluator_loaded": evaluator is not None
    }
    
    if rag_inference:
        info.update({
            "model_path": rag_inference.model_path,
            "index_path": rag_inference.index_path,
            "use_gemini": rag_inference.use_gemini,
            "has_model": rag_inference.model is not None,
            "has_index": rag_inference.indexer is not None
        })
    
    return info


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

