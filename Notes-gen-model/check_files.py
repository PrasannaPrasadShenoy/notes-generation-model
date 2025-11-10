"""
File Verification Script
Checks if all required files for hybrid RAG model are present
"""

import os
from pathlib import Path

def check_files():
    """Check all required files"""
    
    base_dir = Path(__file__).parent
    
    # Required files
    required_files = {
        "Core Training": [
            "train.py",
            "train_custom.py",
            "config.py",
            "utils.py",
            "inference.py",
            "enhanced_notes.py"
        ],
        "RAG Pipeline": [
            "src/rag_inference.py",
            "src/indexing.py",
            "src/ingestion.py",
            "src/train_lora.py",
            "src/evaluate.py",
            "src/utils/prompt_builder.py",
            "src/utils/__init__.py",
            "src/__init__.py"
        ],
        "API Servers": [
            "api_server.py",
            "src/api_rag.py"
        ],
        "Data & Config": [
            "data/transcripts_sample.jsonl",
            "requirements.txt"
        ],
        "Utilities": [
            "prepare_training_data.py",
            "run_rag_pipeline.py"
        ]
    }
    
    # Optional files (generated)
    optional_files = [
        ".env",
        "knowledge_index.index",
        "knowledge_index.metadata",
        "data/processed_chunks.json",
        "ila-notes-generator/config.json"
    ]
    
    print("=" * 60)
    print("File Verification - Hybrid RAG Model")
    print("=" * 60)
    
    all_present = True
    
    # Check required files
    for category, files in required_files.items():
        print(f"\n[{category}]:")
        for file in files:
            file_path = base_dir / file
            exists = file_path.exists()
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {file}")
            if not exists:
                all_present = False
    
    # Check optional files
    print(f"\n[Optional Files (Generated)]:")
    for file in optional_files:
        file_path = base_dir / file
        exists = file_path.exists()
        status = "[OK]" if exists else "[WILL BE CREATED]"
        print(f"  {status} {file}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_present:
        print("[SUCCESS] ALL REQUIRED FILES ARE PRESENT!")
        print("\nYou're ready to:")
        print("  1. Train the model: python train.py")
        print("  2. Set up RAG: python run_rag_pipeline.py")
        print("  3. Use hybrid RAG: python example_hybrid_rag.py")
    else:
        print("[ERROR] SOME REQUIRED FILES ARE MISSING!")
        print("Please check the files marked with [MISSING] above")
    print("=" * 60)
    
    # Check dependencies
    print("\n[Checking Dependencies in requirements.txt]...")
    req_file = base_dir / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"  [OK] Found {len(deps)} dependencies")
        print(f"  Install with: pip install -r requirements.txt")
    else:
        print("  [ERROR] requirements.txt not found")
    
    return all_present

if __name__ == "__main__":
    check_files()

