"""
Example: Using Hybrid RAG with Gemini API
Demonstrates the hybrid approach: Local retrieval + Gemini generation
"""

import os
from src.rag_inference import RAGInference
import json

def example_hybrid_rag():
    """Example of hybrid RAG with Gemini"""
    
    print("=" * 60)
    print("Hybrid RAG + Gemini Example")
    print("=" * 60)
    
    # Initialize RAG (auto-detects Gemini from .env/config)
    rag = RAGInference(
        model_path="./ila-notes-generator",
        index_path="./knowledge_index"
    )
    
    # Check what's available
    print(f"\nConfiguration:")
    print(f"  Gemini Available: {rag.use_gemini and rag.gemini_client is not None}")
    print(f"  Local Model Available: {rag.model is not None}")
    print(f"  Index Available: {rag.indexer is not None}")
    
    # Sample transcript
    transcript = """
    Neural networks are computational models inspired by biological neurons.
    They consist of layers of interconnected nodes, each with adjustable weights.
    The process of training involves forward propagation, where inputs pass through
    the network, and backpropagation, where errors are used to adjust weights.
    Activation functions introduce non-linearity, allowing networks to learn
    complex patterns. Common architectures include feedforward networks,
    convolutional neural networks for image processing, and recurrent neural
    networks for sequential data.
    """
    
    # Generate with auto-detection (uses Gemini if available, else local)
    print("\n" + "=" * 60)
    print("Generating Notes (Auto-detect backend)...")
    print("=" * 60)
    
    notes = rag.generate_notes(
        transcript=transcript,
        query="Explain neural networks for beginners",
        top_k=5
    )
    
    print(f"\n✅ Generated using: {notes.get('generation_method', 'unknown')}")
    print(f"\n📝 Summary:")
    print(notes.get('summary', 'N/A'))
    print(f"\n🧩 Key Concepts: {len(notes.get('key_concepts', []))}")
    print(f"📚 Sources: {len(notes.get('context_sources', []))}")
    
    # Force local model
    if rag.model:
        print("\n" + "=" * 60)
        print("Generating Notes (Forced Local Model)...")
        print("=" * 60)
        
        notes_local = rag.generate_notes(
            transcript=transcript,
            query="Explain neural networks",
            force_local=True
        )
        
        print(f"\n✅ Generated using: {notes_local.get('generation_method', 'unknown')}")
    
    # Force Gemini (if available)
    if rag.use_gemini and rag.gemini_client:
        print("\n" + "=" * 60)
        print("Generating Notes (Forced Gemini API)...")
        print("=" * 60)
        
        notes_gemini = rag.generate_notes(
            transcript=transcript,
            query="Explain neural networks",
            force_gemini=True
        )
        
        print(f"\n✅ Generated using: {notes_gemini.get('generation_method', 'unknown')}")
        print(f"\n📝 Full Response:")
        print(json.dumps(notes_gemini, indent=2)[:500] + "...")


def example_comparison():
    """Compare local vs Gemini output"""
    
    print("\n" + "=" * 60)
    print("Quality Comparison: Local vs Gemini")
    print("=" * 60)
    
    rag = RAGInference(
        model_path="./ila-notes-generator",
        index_path="./knowledge_index"
    )
    
    transcript = "Backpropagation is the algorithm used to train neural networks..."
    
    if rag.model:
        print("\n1. Local Model Output:")
        local_notes = rag.generate_notes(transcript, force_local=True)
        print(f"   Method: {local_notes.get('generation_method')}")
        print(f"   Summary: {local_notes.get('summary', 'N/A')[:100]}...")
    
    if rag.use_gemini and rag.gemini_client:
        print("\n2. Gemini API Output:")
        gemini_notes = rag.generate_notes(transcript, force_gemini=True)
        print(f"   Method: {gemini_notes.get('generation_method')}")
        print(f"   Summary: {gemini_notes.get('summary', 'N/A')[:100]}...")
        print(f"   Key Concepts: {len(gemini_notes.get('key_concepts', []))}")


if __name__ == "__main__":
    try:
        example_hybrid_rag()
        example_comparison()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Model is trained: ./ila-notes-generator exists")
        print("  2. Index is built: ./knowledge_index.index exists")
        print("  3. If using Gemini: Set GEMINI_API_KEY in .env")
        import traceback
        traceback.print_exc()

