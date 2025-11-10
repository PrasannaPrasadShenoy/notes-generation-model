"""
Quick test script for Hybrid RAG model
Run this after training and setting up RAG pipeline
"""

from src.rag_inference import RAGInference
import json

def main():
    print("=" * 60)
    print("Testing Hybrid RAG Model")
    print("=" * 60)
    
    # Initialize RAG
    print("\n1. Loading RAG system...")
    try:
        rag = RAGInference(
            model_path="./ila-notes-generator",
            index_path="./knowledge_index"
        )
        print("✅ RAG system loaded")
        if rag.use_gemini and rag.gemini_client:
            print(f"   Using: Gemini API ({rag.gemini_model})")
        elif rag.model:
            print("   Using: Local model")
        else:
            print("   ⚠️  No generation backend available")
    except Exception as e:
        print(f"❌ Error loading RAG: {e}")
        print("\nMake sure you have:")
        print("  - Trained model: python train.py")
        print("  - Built index: python run_rag_pipeline.py")
        return
    
    # Test transcript
    transcript = """
    Neural networks are computational models inspired by biological neurons.
    They consist of layers of interconnected nodes with adjustable weights.
    Each node receives inputs, applies a transformation, and produces an output.
    Training involves forward propagation and backpropagation algorithms.
    Forward propagation passes data through the network to make predictions.
    Backpropagation adjusts weights to minimize prediction errors.
    """
    
    print("\n2. Generating notes...")
    print(f"   Transcript length: {len(transcript)} characters")
    
    try:
        notes = rag.generate_notes(
            transcript,
            query="Explain neural networks for beginners",
            top_k=5
        )
        
        print("✅ Notes generated successfully")
        print(f"\n3. Results:")
        print("=" * 60)
        print(f"Generation Method: {notes.get('generation_method', 'unknown')}")
        print(f"\n📝 Summary:")
        print(notes.get('summary', 'N/A'))
        print(f"\n🧩 Key Concepts ({len(notes.get('key_concepts', []))}):")
        for i, concept in enumerate(notes.get('key_concepts', [])[:5], 1):
            print(f"   {i}. {concept}")
        print(f"\n📚 Detailed Explanation:")
        print(notes.get('detailed_explanation', 'N/A')[:300] + "...")
        print(f"\n💡 Example:")
        print(notes.get('example', 'N/A')[:200])
        print(f"\n🎯 Prerequisites ({len(notes.get('prerequisites', []))}):")
        for prereq in notes.get('prerequisites', [])[:3]:
            print(f"   - {prereq}")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        
        # Save to file
        with open('test_output.json', 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        print("📄 Full output saved to: test_output.json")
        
    except Exception as e:
        print(f"❌ Error generating notes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

