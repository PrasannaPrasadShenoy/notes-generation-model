"""
ILA Notes Generation Model - Inference Script
Generates enhanced notes from lecture transcripts
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from config import (
    MODEL_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    DEFAULT_MIN_LENGTH, SHORT_NOTES_MAX_LENGTH, SHORT_NOTES_MIN_LENGTH,
    SHORT_NOTES_NUM_BEAMS, SHORT_NOTES_LENGTH_PENALTY,
    DETAILED_NOTES_MAX_LENGTH, DETAILED_NOTES_MIN_LENGTH,
    DETAILED_NOTES_NUM_BEAMS, DETAILED_NOTES_LENGTH_PENALTY
)
from utils import (
    clean_text, format_notes, estimate_reading_time,
    validate_transcript, post_process_notes, extract_key_phrases
)

class NotesGenerator:
    """Notes Generator class for inference"""
    
    def __init__(self, model_dir=MODEL_DIR):
        """Initialize the notes generator"""
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. "
                "Please train the model first using train.py"
            )
        
        print(f"📂 Loading model from {self.model_dir}...")
        print(f"🖥️  Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
            
            # Load training info if available
            info_path = os.path.join(self.model_dir, "training_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    self.training_info = json.load(f)
                    print(f"   Model trained on: {self.training_info.get('dataset', 'unknown')}")
                    print(f"   Training samples: {self.training_info.get('train_samples', 'unknown')}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def generate_notes(
        self,
        text,
        max_length=MAX_TARGET_LENGTH,
        min_length=DEFAULT_MIN_LENGTH,
        num_beams=4,
        length_penalty=2.0,
        temperature=1.0,
        do_sample=False
    ):
        """
        Generate notes from input text
        
        Args:
            text: Input transcript or article text
            max_length: Maximum length of generated notes
            min_length: Minimum length of generated notes
            num_beams: Number of beams for beam search
            length_penalty: Length penalty (higher = longer outputs)
            temperature: Sampling temperature (only if do_sample=True)
            do_sample: Whether to use sampling instead of greedy/beam search
        
        Returns:
            Generated notes as string
        """
        if not text or len(text.strip()) == 0:
            return "Error: Empty input text"
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        # Decode
        generated_notes = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return generated_notes
    
    def generate_short_notes(self, text, format_output=True):
        """
        Generate concise short notes (5-8 key points)
        
        Args:
            text: Input transcript
            format_output: If True, format as bullet points
        
        Returns:
            Formatted short notes
        """
        # Validate input
        is_valid, error = validate_transcript(text)
        if not is_valid:
            return f"Error: {error}"
        
        # Clean text
        text = clean_text(text)
        
        # Generate notes
        notes = self.generate_notes(
            text,
            max_length=SHORT_NOTES_MAX_LENGTH,
            min_length=SHORT_NOTES_MIN_LENGTH,
            num_beams=SHORT_NOTES_NUM_BEAMS,
            length_penalty=SHORT_NOTES_LENGTH_PENALTY
        )
        
        # Post-process
        notes = post_process_notes(notes)
        
        # Format if requested
        if format_output:
            notes = format_notes(notes, format_type="bullet")
        
        return notes
    
    def generate_detailed_notes(self, text, format_output=True, include_metadata=True):
        """
        Generate comprehensive detailed notes with additional insights
        
        Args:
            text: Input transcript
            format_output: If True, format as markdown
            include_metadata: If True, include reading time and key phrases
        
        Returns:
            Detailed notes with optional metadata
        """
        # Validate input
        is_valid, error = validate_transcript(text)
        if not is_valid:
            return f"Error: {error}"
        
        # Clean text
        text = clean_text(text)
        
        # Generate notes
        notes = self.generate_notes(
            text,
            max_length=DETAILED_NOTES_MAX_LENGTH,
            min_length=DETAILED_NOTES_MIN_LENGTH,
            num_beams=DETAILED_NOTES_NUM_BEAMS,
            length_penalty=DETAILED_NOTES_LENGTH_PENALTY
        )
        
        # Post-process
        notes = post_process_notes(notes)
        
        # Format if requested
        if format_output:
            notes = format_notes(notes, format_type="markdown")
        
        # Add metadata if requested
        if include_metadata:
            reading_time = estimate_reading_time(notes)
            key_phrases = extract_key_phrases(text, max_phrases=5)
            
            metadata = f"""
## Notes

{notes}

---
**Reading Time**: ~{reading_time['reading_time_minutes']} minutes ({reading_time['word_count']} words)
**Key Topics**: {', '.join(key_phrases[:5])}
"""
            return metadata.strip()
        
        return notes
    
    def generate_enhanced_notes(self, text):
        """
        Generate enhanced notes that add useful information beyond just summarization
        This method generates detailed notes with additional context and insights
        """
        return self.generate_detailed_notes(text, format_output=True, include_metadata=True)


def generate_notes(text, model_dir=MODEL_DIR, detailed=False, enhanced=False):
    """
    Convenience function to generate notes
    
    Args:
        text: Input transcript or article text
        model_dir: Path to trained model directory
        detailed: If True, generate detailed notes; if False, generate short notes
        enhanced: If True, generate enhanced notes with metadata and insights
    
    Returns:
        Generated notes as string
    """
    generator = NotesGenerator(model_dir)
    
    if enhanced:
        return generator.generate_enhanced_notes(text)
    elif detailed:
        return generator.generate_detailed_notes(text)
    else:
        return generator.generate_short_notes(text)


def main():
    """Main function for testing inference"""
    print("=" * 60)
    print("🧠 ILA Notes Generation Model - Inference")
    print("=" * 60)
    
    # Sample transcript (you can replace this with actual YouTube transcript)
    sample_transcript = """
    Artificial intelligence is transforming education by providing adaptive learning systems 
    that personalize the educational experience for each student. Machine learning algorithms 
    analyze student performance data to identify learning patterns and adjust content delivery 
    accordingly. These systems can detect when a student is struggling with a concept and 
    provide additional resources or alternative explanations. Natural language processing 
    enables AI tutors to answer student questions in real-time, providing 24/7 support. 
    Computer vision technologies can track student engagement and attention, helping educators 
    understand when students are focused or distracted. AI-powered assessment tools can 
    automatically grade assignments and provide instant feedback, freeing up educators to 
    focus on more interactive teaching methods. The integration of AI in education also 
    enables the creation of intelligent content that adapts to different learning styles, 
    whether visual, auditory, or kinesthetic. However, challenges remain, including ensuring 
    data privacy, addressing algorithmic bias, and maintaining the human element in education. 
    As AI continues to evolve, it promises to make education more accessible, efficient, and 
    effective for learners worldwide.
    """
    
    try:
        # Initialize generator
        generator = NotesGenerator()
        
        print("\n📝 Generating Short Notes...")
        print("-" * 60)
        short_notes = generator.generate_short_notes(sample_transcript)
        print(short_notes)
        print("-" * 60)
        
        print("\n📚 Generating Detailed Notes...")
        print("-" * 60)
        detailed_notes = generator.generate_detailed_notes(sample_transcript)
        print(detailed_notes)
        print("-" * 60)
        
        print("\n✨ Generating Enhanced Notes (with metadata)...")
        print("-" * 60)
        enhanced_notes = generator.generate_enhanced_notes(sample_transcript)
        print(enhanced_notes)
        print("-" * 60)
        
        print("\n✅ Inference test complete!")
        print("\n💡 Usage example:")
        print("   from inference import generate_notes")
        print("   notes = generate_notes(your_transcript_text)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please train the model first:")
        print("   python train.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

