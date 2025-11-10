"""
Data Ingestion Module
Cleans transcripts, splits into chunks, and prepares data for indexing
"""

import re
import json
from typing import List, Dict, Optional
from pathlib import Path


class TranscriptIngester:
    """Ingest and preprocess transcripts for RAG"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize ingester
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_transcript(self, text: str) -> str:
        """
        Clean and normalize transcript text
        
        Args:
            text: Raw transcript text
        
        Returns:
            Cleaned text
        """
        # Remove timestamps [00:00] or 00:00
        text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
        
        # Remove speaker labels
        text = re.sub(r'\[?Speaker\s+\d+\]?:?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[?[A-Z][a-z]+\s+[A-Z][a-z]+\]?:?\s*', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove filler words (optional - can be customized)
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        words = text.split()
        words = [w for w in words if w.lower() not in filler_words]
        text = ' '.join(words)
        
        return text.strip()
    
    def split_into_chunks(
        self, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        # Split by sentences first for better chunk boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': start + len(current_chunk),
                    'text': current_chunk.strip()
                }
                if metadata:
                    chunk_metadata.update(metadata)
                chunks.append(chunk_metadata)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                start = start + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = {
                'chunk_id': chunk_id,
                'start_char': start,
                'end_char': start + len(current_chunk),
                'text': current_chunk.strip()
            }
            if metadata:
                chunk_metadata.update(metadata)
            chunks.append(chunk_metadata)
        
        return chunks
    
    def process_transcript(
        self, 
        transcript: str, 
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process a transcript: clean and chunk
        
        Args:
            transcript: Raw transcript text
            metadata: Optional metadata (video_id, title, etc.)
        
        Returns:
            List of processed chunks
        """
        cleaned = self.clean_transcript(transcript)
        chunks = self.split_into_chunks(cleaned, metadata)
        return chunks
    
    def process_file(
        self, 
        file_path: str, 
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a transcript file
        
        Args:
            file_path: Path to transcript file
            output_path: Optional path to save processed chunks
        
        Returns:
            List of processed chunks
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'source_file': file_path,
            'source_type': 'file'
        }
        
        chunks = self.process_transcript(content, metadata)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        return chunks
    
    def process_jsonl(
        self, 
        jsonl_path: str, 
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a JSONL file with multiple transcripts
        
        Args:
            jsonl_path: Path to JSONL file
            output_path: Optional path to save processed chunks
        
        Returns:
            List of all processed chunks
        """
        all_chunks = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    transcript = data.get('transcript', '')
                    metadata = {
                        'topic': data.get('topic', ''),
                        'reference': data.get('reference', ''),
                        'source_type': 'jsonl',
                        'line_number': line_num
                    }
                    
                    chunks = self.process_transcript(transcript, metadata)
                    all_chunks.extend(chunks)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        return all_chunks


def main():
    """Example usage"""
    ingester = TranscriptIngester(chunk_size=500, chunk_overlap=50)
    
    # Example transcript
    sample_transcript = """
    [00:00] Neural networks are computational models inspired by biological neurons.
    [00:15] They consist of layers of interconnected nodes, each with adjustable weights.
    [00:30] The process of training involves backpropagation and gradient descent.
    """
    
    chunks = ingester.process_transcript(sample_transcript, {'topic': 'Neural Networks'})
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}: {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()

