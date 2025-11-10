"""
RAG Inference Module
Combines retrieval and generation for context-aware note generation
"""

import os
import json
import re
from typing import Dict, List, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
try:
    from src.indexing import VectorIndexer
except ImportError:
    # Fallback for direct import
    from indexing import VectorIndexer
from src.utils.prompt_builder import build_gemini_prompt, build_local_prompt, extract_json_from_response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import USE_GEMINI_API, GEMINI_API_KEY, GEMINI_MODEL, TOP_K_RETRIEVAL
except ImportError:
    USE_GEMINI_API = os.getenv("USE_GEMINI_API", "false").lower() == "true"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))


class RAGInference:
    """RAG-based inference for generating insightful notes"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        use_gemini: Optional[bool] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None
    ):
        """
        Initialize RAG inference with hybrid support
        
        Args:
            model_path: Path to fine-tuned model (LoRA or full)
            index_path: Path to FAISS index
            use_gemini: Whether to use Gemini API (auto-detect from config if None)
            gemini_api_key: Gemini API key (or from env/config)
            gemini_model: Gemini model name (default: gemini-1.5-pro)
        """
        self.model_path = model_path
        self.index_path = index_path
        
        # Gemini configuration
        self.use_gemini = use_gemini if use_gemini is not None else USE_GEMINI_API
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or GEMINI_MODEL
        self.top_k = TOP_K_RETRIEVAL
        
        self.model = None
        self.tokenizer = None
        self.indexer = None
        self.is_seq2seq = True
        self.gemini_client = None
        
        # Initialize Gemini if enabled
        if self.use_gemini and self.gemini_api_key:
            self._init_gemini()
        
        # Load components
        if index_path and os.path.exists(f"{index_path}.index"):
            self.load_index(index_path)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _init_gemini(self):
        """Initialize Gemini API client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel(self.gemini_model)
            print(f"✅ Gemini API initialized (model: {self.gemini_model})")
        except ImportError:
            print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")
            self.use_gemini = False
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            self.use_gemini = False
    
    def load_index(self, index_path: str):
        """Load FAISS index"""
        self.indexer = VectorIndexer()
        self.indexer.load_index(index_path)
        print("✅ Index loaded")
    
    def load_model(self, model_path: str):
        """Load fine-tuned model"""
        print(f"Loading model from {model_path}")
        
        # Check if it's a LoRA model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Load base model
            base_model_name = "google/flan-t5-large"  # Default, can be configured
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.is_seq2seq = True
        else:
            # Regular model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.is_seq2seq = True
            except:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.is_seq2seq = False
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        print("✅ Model loaded")
    
    def retrieve_context(
        self,
        query: str,
        transcript: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant context from index
        
        Args:
            query: User query or topic
            transcript: Lecture transcript
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks
        """
        if not self.indexer:
            return []
        
        # Combine query and transcript for retrieval
        search_text = f"{query} {transcript[:500]}"  # Use first 500 chars of transcript
        
        results = self.indexer.query_index(search_text, top_k=top_k)
        return results
    
    def build_prompt(
        self,
        transcript: str,
        context_chunks: List[Dict],
        query: Optional[str] = None,
        use_gemini_format: bool = None
    ) -> str:
        """
        Build prompt with context (uses appropriate format for Gemini or local model)
        
        Args:
            transcript: Lecture transcript
            context_chunks: Retrieved context chunks
            query: Optional user query
            use_gemini_format: Whether to use Gemini format (auto-detect if None)
        
        Returns:
            Formatted prompt
        """
        if use_gemini_format is None:
            use_gemini_format = self.use_gemini and self.gemini_client is not None
        
        if use_gemini_format:
            return build_gemini_prompt(transcript, context_chunks, query)
        else:
            return build_local_prompt(transcript, context_chunks, query)
    
    def generate_with_model(self, prompt: str) -> str:
        """Generate using local model"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    min_length=100,
                    num_beams=4,
                    length_penalty=2.0,
                    temperature=0.7,
                    do_sample=True
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def generate_with_gemini(self, prompt: str) -> Dict:
        """
        Generate using Gemini API with structured JSON output
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Dictionary with generated content
        """
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")
        
        try:
            # Generate with Gemini
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            
            generated_text = response.text
            
            # Try to extract JSON
            try:
                structured = extract_json_from_response(generated_text)
                return structured
            except Exception as e:
                print(f"Warning: Could not parse JSON from Gemini response: {e}")
                # Fallback to text parsing
                return self.parse_structured_output(generated_text)
                
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            raise
    
    def parse_structured_output(self, text: str) -> Dict:
        """
        Parse generated text into structured format
        
        Args:
            text: Generated text
        
        Returns:
            Structured dictionary
        """
        # Try to extract structured sections
        result = {
            "summary": "",
            "key_concepts": [],
            "detailed_explanation": "",
            "example": "",
            "prerequisites": [],
            "further_reading": [],
            "sources": []
        }
        
        # Simple parsing (can be improved with regex or LLM)
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'summary' in line_lower or 'overview' in line_lower:
                current_section = 'summary'
            elif 'key concept' in line_lower or 'concept' in line_lower:
                current_section = 'key_concepts'
            elif 'explanation' in line_lower or 'detailed' in line_lower:
                current_section = 'detailed_explanation'
            elif 'example' in line_lower or 'application' in line_lower:
                current_section = 'example'
            elif 'prerequisite' in line_lower:
                current_section = 'prerequisites'
            elif 'further reading' in line_lower or 'resource' in line_lower:
                current_section = 'further_reading'
            elif 'source' in line_lower or 'citation' in line_lower:
                current_section = 'sources'
            
            if current_section and line.strip():
                if current_section == 'key_concepts' or current_section == 'prerequisites' or current_section == 'further_reading':
                    if line.strip().startswith('-') or line.strip().startswith('*') or line.strip()[0].isdigit():
                        result[current_section].append(line.strip().lstrip('-* ').lstrip('0123456789. '))
                else:
                    if result[current_section]:
                        result[current_section] += " " + line.strip()
                    else:
                        result[current_section] = line.strip()
        
        # Fallback: if parsing failed, put everything in detailed_explanation
        if not any(result.values()):
            result['detailed_explanation'] = text
            # Try to extract summary (first 2-3 sentences)
            sentences = text.split('. ')
            result['summary'] = '. '.join(sentences[:3]) + '.'
        
        return result
    
    def generate_notes(
        self,
        transcript: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        use_structured: bool = True,
        force_gemini: bool = False,
        force_local: bool = False
    ) -> Dict:
        """
        Generate insightful notes using Hybrid RAG (Gemini + Local Retrieval)
        
        Args:
            transcript: Lecture transcript
            query: Optional query or topic focus
            top_k: Number of context chunks to retrieve (default: from config)
            use_structured: Whether to return structured format
            force_gemini: Force use of Gemini (even if not configured)
            force_local: Force use of local model (even if Gemini available)
        
        Returns:
            Generated notes (structured dictionary)
        """
        # Determine which generator to use
        use_gemini = (force_gemini or (self.use_gemini and self.gemini_client)) and not force_local
        
        # Retrieve context (always local, free and fast)
        top_k = top_k or self.top_k
        context_chunks = self.retrieve_context(
            query or "explain this topic", 
            transcript, 
            top_k
        )
        
        # Build appropriate prompt
        prompt = self.build_prompt(
            transcript, 
            context_chunks, 
            query,
            use_gemini_format=use_gemini
        )
        
        # Generate with appropriate backend
        try:
            if use_gemini:
                # Use Gemini API (high quality)
                result = self.generate_with_gemini(prompt)
                result['generation_method'] = 'gemini'
            elif self.model:
                # Use local model (free, private)
                generated_text = self.generate_with_model(prompt)
                result = self.parse_structured_output(generated_text)
                result['raw_output'] = generated_text
                result['generation_method'] = 'local'
            else:
                raise ValueError("No model or API available for generation")
        except Exception as e:
            # Fallback to local model if Gemini fails
            if use_gemini and self.model:
                print(f"⚠️  Gemini failed ({e}), falling back to local model")
                generated_text = self.generate_with_model(prompt)
                result = self.parse_structured_output(generated_text)
                result['raw_output'] = generated_text
                result['generation_method'] = 'local_fallback'
            else:
                raise
        
        # Add metadata
        if use_structured:
            result['context_sources'] = [
                {
                    'id': i,
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'score': float(chunk.get('score', 0)),
                    'source': chunk.get('reference', chunk.get('source_file', chunk.get('topic', 'unknown')))
                }
                for i, chunk in enumerate(context_chunks)
            ]
            result['retrieval_count'] = len(context_chunks)
            result['query'] = query
            result['transcript_length'] = len(transcript)
        
        return result


def main():
    """Example usage"""
    # Initialize RAG
    rag = RAGInference(
        model_path="./ila-notes-generator",  # Your trained model
        index_path="./knowledge_index",  # Your FAISS index
        use_gemini=False  # Set to True to use Gemini API
    )
    
    # Generate notes
    transcript = "Neural networks are computational models inspired by biological neurons..."
    notes = rag.generate_notes(transcript, query="Explain neural networks")
    
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()

