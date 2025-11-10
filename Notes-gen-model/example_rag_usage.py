"""
Example: Using RAG System for Enhanced Note Generation
Shows how to use the new RAG pipeline alongside existing functionality
"""

import json
from inference import generate_notes, NotesGenerator
from src.rag_inference import RAGInference
from src.evaluate import NoteEvaluator


def example_basic_generation():
    """Example 1: Basic generation (existing functionality)"""
    print("=" * 60)
    print("Example 1: Basic Note Generation")
    print("=" * 60)
    
    transcript = """
    Neural networks are computational models inspired by biological neurons.
    They consist of layers of interconnected nodes, each with adjustable weights.
    The process of training involves forward propagation and backpropagation.
    """
    
    # Use existing generator
    notes = generate_notes(transcript, enhanced=True)
    print(notes)
    print()


def example_rag_generation():
    """Example 2: RAG-based generation (new functionality)"""
    print("=" * 60)
    print("Example 2: RAG-Based Note Generation")
    print("=" * 60)
    
    transcript = """
    Neural networks are computational models inspired by biological neurons.
    They consist of layers of interconnected nodes, each with adjustable weights.
    """
    
    try:
        # Initialize RAG (requires index to be built first)
        rag = RAGInference(
            model_path="./ila-notes-generator",
            index_path="./knowledge_index"  # Must exist
        )
        
        # Generate with context
        notes = rag.generate_notes(
            transcript=transcript,
            query="Explain neural networks",
            top_k=5
        )
        
        print(json.dumps(notes, indent=2))
    except Exception as e:
        print(f"RAG not available: {e}")
        print("Build index first: see RAG_SETUP_GUIDE.md")
    print()


def example_hybrid_approach():
    """Example 3: Hybrid approach - use best available"""
    print("=" * 60)
    print("Example 3: Hybrid Approach")
    print("=" * 60)
    
    transcript = """
    Backpropagation is the algorithm used to train neural networks.
    It computes gradients of the loss function with respect to weights.
    """
    
    # Try RAG first, fallback to basic
    try:
        rag = RAGInference(
            model_path="./ila-notes-generator",
            index_path="./knowledge_index"
        )
        notes = rag.generate_notes(transcript, query="explain backpropagation")
        print("✅ Using RAG (context-aware)")
        print(json.dumps(notes, indent=2))
    except:
        print("⚠️ RAG not available, using basic generation")
        notes = generate_notes(transcript, enhanced=True)
        print(notes)
    print()


def example_evaluation():
    """Example 4: Evaluate note quality"""
    print("=" * 60)
    print("Example 4: Note Evaluation")
    print("=" * 60)
    
    transcript = "Neural networks are computational models..."
    generated = generate_notes(transcript, enhanced=True)
    
    # Convert to dict format for evaluation
    generated_dict = {
        'raw_output': generated,
        'summary': generated.split('\n')[0] if '\n' in generated else generated[:200],
        'detailed_explanation': generated
    }
    
    reference = "Neural networks are AI models that mimic biological neurons..."
    
    evaluator = NoteEvaluator()
    metrics = evaluator.evaluate_notes(
        generated=generated_dict,
        reference=reference,
        transcript=transcript
    )
    
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    print()


def example_complete_pipeline():
    """Example 5: Complete pipeline from transcript to evaluated notes"""
    print("=" * 60)
    print("Example 5: Complete Pipeline")
    print("=" * 60)
    
    transcript = """
    Convolutional Neural Networks, or CNNs, are specialized neural networks
    for processing grid-like data such as images. They use convolutional layers
    that apply filters to detect features like edges and patterns.
    """
    
    # Step 1: Generate notes
    print("Step 1: Generating notes...")
    notes = generate_notes(transcript, enhanced=True)
    
    # Step 2: Evaluate (if reference available)
    print("\nStep 2: Evaluating quality...")
    evaluator = NoteEvaluator()
    generated_dict = {'raw_output': notes, 'detailed_explanation': notes}
    metrics = evaluator.evaluate_notes(generated_dict, transcript=transcript)
    
    print(f"Similarity: {metrics.get('transcript_similarity', 'N/A'):.3f}")
    print(f"Structure: {metrics.get('structure_completeness', 'N/A'):.3f}")
    
    # Step 3: Display notes
    print("\nStep 3: Generated Notes:")
    print(notes)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ILA Insight Generator - Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_generation()
    example_rag_generation()
    example_hybrid_approach()
    example_evaluation()
    example_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ Examples complete!")
    print("=" * 60)
    print("\nFor more details, see:")
    print("- README_RAG.md - RAG system documentation")
    print("- RAG_SETUP_GUIDE.md - Setup instructions")
    print("- ENHANCEMENTS_SUMMARY.md - What was added")

