#!/usr/bin/env python3
"""
Complete RAG Pipeline Setup and Test
Run this to set up and test the RAG pipeline
"""

import os
import json
from pathlib import Path

def check_prerequisites():
    """Check if prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check if model exists
    if not os.path.exists("./ila-notes-generator"):
        print("❌ Model not found at ./ila-notes-generator")
        print("   Run 'python train.py' first")
        return False
    
    # Check if data exists
    if not os.path.exists("data/transcripts_sample.jsonl"):
        print("⚠️  Sample data not found, but we can create it")
    
    print("✅ Prerequisites check complete")
    return True

def setup_pipeline():
    """Set up the complete RAG pipeline"""
    
    print("\n" + "="*60)
    print("Setting up RAG Pipeline")
    print("="*60 + "\n")
    
    # Step 1: Process transcripts
    print("Step 1: Processing transcripts...")
    try:
        from src.ingestion import TranscriptIngester
        
        ingester = TranscriptIngester(chunk_size=500, chunk_overlap=50)
        
        if os.path.exists("data/transcripts_sample.jsonl"):
            ingester.process_jsonl(
                'data/transcripts_sample.jsonl',
                'data/processed_chunks.json'
            )
            print("✅ Transcripts processed")
        else:
            print("⚠️  No transcripts found, creating sample...")
            # Create sample
            os.makedirs("data", exist_ok=True)
            sample_data = [
                {
                    "topic": "Neural Networks",
                    "transcript": "Neural networks are computational models inspired by biological neurons. They consist of layers of interconnected nodes with adjustable weights.",
                    "reference": "Sample Lecture"
                }
            ]
            with open("data/transcripts_sample.jsonl", "w") as f:
                for item in sample_data:
                    f.write(json.dumps(item) + "\n")
            ingester.process_jsonl(
                'data/transcripts_sample.jsonl',
                'data/processed_chunks.json'
            )
            print("✅ Sample transcripts created and processed")
    except Exception as e:
        print(f"❌ Error processing transcripts: {e}")
        return False
    
    # Step 2: Build index
    print("\nStep 2: Building vector index...")
    try:
        from src.indexing import build_index_from_data
        
        if not os.path.exists("data/processed_chunks.json"):
            print("❌ Processed chunks not found")
            return False
        
        build_index_from_data(
            'data/processed_chunks.json',
            'knowledge_index'
        )
        print("✅ Vector index built")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return False
    
    # Step 3: Test RAG
    print("\nStep 3: Testing RAG generation...")
    try:
        from src.rag_inference import RAGInference
        
        rag = RAGInference(
            model_path="./ila-notes-generator",
            index_path="./knowledge_index"
        )
        
        test_transcript = """
        Neural networks are computational models inspired by biological neurons.
        They consist of layers of interconnected nodes, each with adjustable weights.
        The process of training involves forward propagation and backpropagation.
        """
        
        print("   Generating test notes...")
        notes = rag.generate_notes(
            test_transcript,
            query="Explain neural networks",
            top_k=3
        )
        
        print("✅ RAG pipeline working!")
        print("\n   Sample output:")
        if isinstance(notes, dict):
            print(f"   Summary: {notes.get('summary', 'N/A')[:100]}...")
            print(f"   Key Concepts: {len(notes.get('key_concepts', []))} concepts")
        else:
            print(f"   {str(notes)[:200]}...")
        
        return rag
        
    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        print("   Note: This might be okay if index is empty")
        return None

def main():
    """Main function"""
    print("\n" + "="*60)
    print("ILA RAG Pipeline Setup")
    print("="*60)
    
    if not check_prerequisites():
        return
    
    rag = setup_pipeline()
    
    if rag:
        print("\n" + "="*60)
        print("✅ RAG Pipeline Ready!")
        print("="*60)
        print("\nUsage:")
        print("  from src.rag_inference import RAGInference")
        print("  rag = RAGInference(model_path='./ila-notes-generator', index_path='./knowledge_index')")
        print("  notes = rag.generate_notes(transcript, query='your query')")
        print("\nOr start the API server:")
        print("  python src/api_rag.py")
    else:
        print("\n⚠️  Setup completed with warnings. Check errors above.")

if __name__ == "__main__":
    main()

