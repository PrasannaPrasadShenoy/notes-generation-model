"""
Evaluation Module
Evaluate note generation quality using ROUGE, similarity metrics, etc.
"""

import json
from typing import Dict, List, Optional
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np


class NoteEvaluator:
    """Evaluate quality of generated notes"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def compute_rouge(self, generated: str, reference: str) -> Dict:
        """
        Compute ROUGE scores
        
        Args:
            generated: Generated text
            reference: Reference text
        
        Returns:
            ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def evaluate_notes(
        self,
        generated: Dict,
        reference: Optional[str] = None,
        transcript: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive evaluation of generated notes
        
        Args:
            generated: Generated notes dictionary
            reference: Optional reference notes
            transcript: Original transcript
        
        Returns:
            Evaluation metrics
        """
        metrics = {}
        
        # Extract generated text
        generated_text = generated.get('raw_output', '')
        if not generated_text:
            # Combine structured fields
            generated_text = f"{generated.get('summary', '')} {generated.get('detailed_explanation', '')}"
        
        # ROUGE scores (if reference provided)
        if reference:
            rouge_scores = self.compute_rouge(generated_text, reference)
            metrics['rouge'] = rouge_scores
            metrics['rouge_avg_f1'] = (
                rouge_scores['rouge1']['fmeasure'] +
                rouge_scores['rouge2']['fmeasure'] +
                rouge_scores['rougeL']['fmeasure']
            ) / 3
        
        # Semantic similarity with transcript
        if transcript:
            similarity = self.compute_semantic_similarity(generated_text, transcript)
            metrics['transcript_similarity'] = similarity
        
        # Structure completeness
        required_fields = ['summary', 'key_concepts', 'detailed_explanation']
        present_fields = [field for field in required_fields if generated.get(field)]
        metrics['structure_completeness'] = len(present_fields) / len(required_fields)
        
        # Length metrics
        metrics['generated_length'] = len(generated_text)
        metrics['word_count'] = len(generated_text.split())
        
        # Key concepts count
        if 'key_concepts' in generated:
            metrics['num_key_concepts'] = len(generated['key_concepts'])
        
        # Context usage
        if 'context_sources' in generated:
            metrics['num_context_sources'] = len(generated['context_sources'])
            if generated['context_sources']:
                avg_score = sum(s.get('score', 0) for s in generated['context_sources']) / len(generated['context_sources'])
                metrics['avg_context_relevance'] = avg_score
        
        return metrics
    
    def batch_evaluate(
        self,
        generated_list: List[Dict],
        reference_list: Optional[List[str]] = None,
        transcript_list: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate multiple generated notes
        
        Args:
            generated_list: List of generated notes
            reference_list: Optional list of references
            transcript_list: Optional list of transcripts
        
        Returns:
            Aggregate metrics
        """
        all_metrics = []
        
        for i, generated in enumerate(generated_list):
            reference = reference_list[i] if reference_list and i < len(reference_list) else None
            transcript = transcript_list[i] if transcript_list and i < len(transcript_list) else None
            
            metrics = self.evaluate_notes(generated, reference, transcript)
            all_metrics.append(metrics)
        
        # Aggregate
        aggregate = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics if key in m]
                aggregate[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            elif isinstance(all_metrics[0][key], dict):
                # For nested dicts like rouge
                aggregate[key] = {}
                for subkey in all_metrics[0][key].keys():
                    if isinstance(all_metrics[0][key][subkey], (int, float)):
                        values = [m[key][subkey] for m in all_metrics if key in m and subkey in m[key]]
                        aggregate[key][subkey] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
        
        return {
            'individual': all_metrics,
            'aggregate': aggregate
        }


def main():
    """Example evaluation"""
    evaluator = NoteEvaluator()
    
    generated = {
        'summary': 'Neural networks are computational models.',
        'key_concepts': ['Neuron', 'Weight', 'Activation'],
        'detailed_explanation': 'Neural networks consist of layers...',
        'raw_output': 'Full generated text here...'
    }
    
    reference = 'Neural networks are AI models that mimic the brain...'
    transcript = 'Neural networks are computational models inspired by biological neurons...'
    
    metrics = evaluator.evaluate_notes(generated, reference, transcript)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

