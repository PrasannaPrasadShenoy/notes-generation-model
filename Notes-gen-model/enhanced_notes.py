"""
Enhanced Notes Generation with Additional Features
Adds key concepts, study tips, related topics, and more to make notes more useful
"""

import re
from typing import List, Dict, Optional
from inference import NotesGenerator
from utils import extract_key_phrases, estimate_reading_time, format_notes

class EnhancedNotesGenerator(NotesGenerator):
    """Extended NotesGenerator with additional enhancement features"""
    
    def __init__(self, model_dir="./ila-notes-generator"):
        super().__init__(model_dir)
    
    def extract_key_concepts(self, text: str, max_concepts: int = 8) -> List[Dict[str, any]]:
        """
        Extract key concepts with context
        
        Returns:
            List of concepts with definitions/explanations
        """
        # Extract key phrases
        phrases = extract_key_phrases(text, max_phrases=max_concepts * 2)
        
        # Find sentences containing these phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        concepts = []
        
        for phrase in phrases[:max_concepts]:
            # Find sentence containing this phrase
            for sentence in sentences:
                if phrase.lower() in sentence.lower():
                    # Clean up sentence
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:
                        concepts.append({
                            'term': phrase.title(),
                            'explanation': sentence[:150] + '...' if len(sentence) > 150 else sentence
                        })
                        break
        
        return concepts[:max_concepts]
    
    def generate_study_tips(self, notes: str, transcript: str) -> List[str]:
        """
        Generate study tips based on content
        
        Returns:
            List of study tips
        """
        tips = []
        
        # Analyze content length
        word_count = len(notes.split())
        if word_count > 300:
            tips.append("📚 This is a comprehensive topic - break it into smaller sections for better retention")
        elif word_count < 100:
            tips.append("💡 This is a concise topic - review it multiple times for reinforcement")
        
        # Check for technical terms
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', transcript)
        if len(technical_terms) > 10:
            tips.append("🔬 Contains many technical terms - create flashcards for key definitions")
        
        # Check for numbers/data
        numbers = re.findall(r'\b\d+\.?\d*\b', transcript)
        if len(numbers) > 5:
            tips.append("📊 Contains numerical data - practice recalling specific figures")
        
        # Check for examples
        if 'example' in transcript.lower() or 'for instance' in transcript.lower():
            tips.append("💭 Review the examples provided - they illustrate key concepts")
        
        # General tips
        tips.append("⏰ Review these notes within 24 hours for better retention")
        tips.append("✍️ Try to explain these concepts in your own words")
        
        return tips[:5]  # Return top 5 tips
    
    def identify_related_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Identify related topics that might be worth exploring
        
        Returns:
            List of related topic suggestions
        """
        # Common topic patterns
        topic_patterns = [
            r'\b(?:introduction to|basics of|fundamentals of|overview of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(?:related to|similar to|like|such as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+(?:a|an|the)',
        ]
        
        topics = set()
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                if len(match) > 3 and len(match) < 50:
                    topics.add(match.strip())
        
        # Also use key phrases
        key_phrases = extract_key_phrases(text, max_phrases=max_topics * 2)
        topics.update([p.title() for p in key_phrases[:max_topics]])
        
        return list(topics)[:max_topics]
    
    def generate_learning_objectives(self, notes: str) -> List[str]:
        """
        Extract or infer learning objectives from notes
        
        Returns:
            List of learning objectives
        """
        objectives = []
        
        # Look for explicit objectives
        obj_patterns = [
            r'(?:you will|students will|learners will|you should|you can)\s+(?:learn|understand|know|be able to)\s+([^.!?]+)',
            r'(?:objective|goal|aim)\s+[:\-]?\s*([^.!?]+)',
        ]
        
        for pattern in obj_patterns:
            matches = re.findall(pattern, notes, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if len(match) > 10 and len(match) < 150:
                    objectives.append(match.capitalize())
        
        # If no explicit objectives found, infer from key points
        if not objectives:
            sentences = notes.split('. ')
            for sentence in sentences[:5]:
                if len(sentence) > 20:
                    # Convert statement to objective
                    obj = sentence.strip()
                    if not obj.endswith('.'):
                        obj += '.'
                    objectives.append(obj)
        
        return objectives[:5]
    
    def generate_enhanced_notes_v2(
        self, 
        transcript: str,
        include_concepts: bool = True,
        include_tips: bool = True,
        include_topics: bool = True,
        include_objectives: bool = True
    ) -> Dict[str, any]:
        """
        Generate comprehensive enhanced notes with all features
        
        Returns:
            Dictionary with notes and all enhancements
        """
        # Generate base notes
        base_notes = self.generate_detailed_notes(
            transcript,
            format_output=True,
            include_metadata=False
        )
        
        # Extract enhancements
        enhancements = {}
        
        if include_concepts:
            enhancements['key_concepts'] = self.extract_key_concepts(transcript)
        
        if include_tips:
            enhancements['study_tips'] = self.generate_study_tips(base_notes, transcript)
        
        if include_topics:
            enhancements['related_topics'] = self.identify_related_topics(transcript)
        
        if include_objectives:
            enhancements['learning_objectives'] = self.generate_learning_objectives(base_notes)
        
        # Get metadata
        reading_time = estimate_reading_time(base_notes)
        key_topics = extract_key_phrases(transcript, max_phrases=10)
        
        # Format output
        formatted_notes = f"""# Study Notes

## Learning Objectives
{chr(10).join([f"- {obj}" for obj in enhancements.get('learning_objectives', [])])}

## Notes

{base_notes}

## Key Concepts

{chr(10).join([f"**{concept['term']}**: {concept['explanation']}" for concept in enhancements.get('key_concepts', [])])}

## Study Tips

{chr(10).join(enhancements.get('study_tips', []))}

## Related Topics to Explore

{chr(10).join([f"- {topic}" for topic in enhancements.get('related_topics', [])])}

---

**Reading Time**: ~{reading_time['reading_time_minutes']} minutes ({reading_time['word_count']} words)  
**Key Topics**: {', '.join(key_topics[:10])}
"""
        
        return {
            'notes': formatted_notes,
            'base_notes': base_notes,
            'enhancements': enhancements,
            'metadata': {
                'reading_time_minutes': reading_time['reading_time_minutes'],
                'word_count': reading_time['word_count'],
                'key_topics': key_topics[:10]
            }
        }
    
    def generate_quick_reference(self, transcript: str) -> str:
        """
        Generate a quick reference card (one-page summary)
        
        Returns:
            Formatted quick reference
        """
        # Generate short notes
        notes = self.generate_short_notes(transcript, format_output=True)
        
        # Extract key concepts (fewer for quick reference)
        concepts = self.extract_key_concepts(transcript, max_concepts=5)
        
        # Get key topics
        topics = extract_key_phrases(transcript, max_phrases=5)
        
        quick_ref = f"""# Quick Reference

## Key Points
{notes}

## Essential Terms
{chr(10).join([f"- **{c['term']}**: {c['explanation'][:80]}..." for c in concepts])}

## Topics Covered
{chr(10).join([f"- {t.title()}" for t in topics])}
"""
        
        return quick_ref


def generate_enhanced_notes_v2(transcript: str, **kwargs) -> Dict[str, any]:
    """
    Convenience function for enhanced notes v2
    
    Args:
        transcript: Input transcript
        **kwargs: Additional options (include_concepts, include_tips, etc.)
    
    Returns:
        Dictionary with enhanced notes
    """
    generator = EnhancedNotesGenerator()
    return generator.generate_enhanced_notes_v2(transcript, **kwargs)

