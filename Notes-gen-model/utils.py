"""
Utility functions for ILA Notes Generation Model
"""

import re
import torch
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}\-]', '', text)
    return text.strip()


def split_long_text(text: str, max_length: int = 1000) -> List[str]:
    """Split long text into chunks that fit within token limits"""
    # Simple sentence-based splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def format_notes(notes: str, format_type: str = "plain") -> str:
    """
    Format generated notes for better readability
    
    Args:
        notes: Raw generated notes
        format_type: "plain", "bullet", "numbered", "markdown"
    """
    if format_type == "bullet":
        # Convert to bullet points
        lines = notes.split('. ')
        formatted = '\n'.join([f"• {line.strip()}" for line in lines if line.strip()])
        return formatted
    
    elif format_type == "numbered":
        # Convert to numbered list
        lines = notes.split('. ')
        formatted = '\n'.join([f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip()])
        return formatted
    
    elif format_type == "markdown":
        # Convert to markdown format
        lines = notes.split('. ')
        formatted = '\n'.join([f"- {line.strip()}" for line in lines if line.strip()])
        return formatted
    
    else:  # plain
        return notes


def estimate_reading_time(text: str, words_per_minute: int = 200) -> Dict[str, float]:
    """Estimate reading time for notes"""
    word_count = len(text.split())
    reading_time_minutes = word_count / words_per_minute
    reading_time_seconds = reading_time_minutes * 60
    
    return {
        "word_count": word_count,
        "reading_time_minutes": round(reading_time_minutes, 1),
        "reading_time_seconds": round(reading_time_seconds, 0)
    }


def check_model_requirements() -> Dict[str, bool]:
    """Check if system meets model requirements"""
    checks = {
        "python_version": True,  # Assume OK if running
        "torch_available": torch.__version__ is not None,
        "cuda_available": torch.cuda.is_available(),
        "transformers_available": True,  # Assume OK if imported
    }
    
    if torch.cuda.is_available():
        checks["cuda_version"] = torch.version.cuda
        checks["gpu_name"] = torch.cuda.get_device_name(0)
        checks["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    
    return checks


def validate_transcript(transcript: str, min_length: int = 100):
    """
    Validate transcript before processing
    
    Returns:
        (is_valid, error_message)
    """
    if not transcript or len(transcript.strip()) == 0:
        return False, "Transcript is empty"
    
    if len(transcript) < min_length:
        return False, f"Transcript too short (minimum {min_length} characters)"
    
    # Check for reasonable content (not just whitespace/special chars)
    word_count = len(transcript.split())
    if word_count < 10:
        return False, "Transcript has too few words"
    
    return True, None


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract potential key phrases from text (simple implementation)"""
    # Simple keyword extraction based on frequency
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    word_freq = {}
    
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_phrases]]


def post_process_notes(notes: str) -> str:
    """Post-process generated notes for better quality"""
    # Remove repeated sentences
    sentences = notes.split('. ')
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if sentence_lower and sentence_lower not in seen:
            seen.add(sentence_lower)
            unique_sentences.append(sentence.strip())
    
    # Join and clean
    processed = '. '.join(unique_sentences)
    if not processed.endswith('.'):
        processed += '.'
    
    return processed

