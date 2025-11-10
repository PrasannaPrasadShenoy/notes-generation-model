"""
Vector Indexing Module
Creates embeddings and FAISS index for semantic search
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss


class VectorIndexer:
    """Create and manage FAISS vector index for semantic search"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu"
    ):
        """
        Initialize indexer
        
        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
        
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings
    
    def build_index(
        self, 
        chunks: List[Dict],
        index_path: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            index_path: Optional path to save index
        
        Returns:
            FAISS index
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index (L2 distance, but normalized = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = chunks
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
        
        if index_path:
            self.save_index(index_path)
        
        return self.index
    
    def query_index(
        self, 
        query_text: str, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Query the index for similar chunks
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of relevant chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query embedding
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['similarity'] = float(score)  # Cosine similarity
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str):
        """
        Save index and metadata to disk
        
        Args:
            index_path: Base path for saving (will create .index and .metadata files)
        """
        index_file = f"{index_path}.index"
        metadata_file = f"{index_path}.metadata"
        config_file = f"{index_path}.config"
        
        # Save FAISS index
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_vectors': self.index.ntotal
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """
        Load index and metadata from disk
        
        Args:
            index_path: Base path for loading
        """
        index_file = f"{index_path}.index"
        metadata_file = f"{index_path}.metadata"
        config_file = f"{index_path}.config"
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Index loaded: {config['num_vectors']} vectors")
        return self.index


def build_index_from_data(
    data_path: str,
    index_path: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
):
    """
    Convenience function to build index from data file
    
    Args:
        data_path: Path to JSON/JSONL file with chunks
        index_path: Path to save index
        model_name: Embedding model name
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            chunks = [json.loads(line) for line in f]
        else:
            chunks = json.load(f)
    
    # Build index
    indexer = VectorIndexer(model_name=model_name)
    indexer.build_index(chunks, index_path)
    
    return indexer


if __name__ == "__main__":
    # Example usage
    from ingestion import TranscriptIngester
    
    # Process sample transcript
    ingester = TranscriptIngester()
    sample_text = "Neural networks are computational models. They use backpropagation for training."
    chunks = ingester.process_transcript(sample_text, {'topic': 'ML'})
    
    # Build index
    indexer = VectorIndexer()
    indexer.build_index(chunks, "test_index")
    
    # Query
    results = indexer.query_index("How do neural networks learn?", top_k=3)
    for r in results:
        print(f"Score: {r['score']:.3f} - {r['text'][:100]}...")

