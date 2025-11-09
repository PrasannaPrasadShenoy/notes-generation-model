"""
Prepare YouTube Transcript Data for Fine-tuning
Converts YouTube transcripts into training format for the notes generation model
"""

import json
import os
from typing import List, Dict, Optional
from datasets import Dataset, DatasetDict
import re

class TranscriptDataPreparer:
    """Prepare YouTube transcript data for training"""
    
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_transcripts_from_json(self, json_file: str) -> List[Dict]:
        """
        Load transcripts from JSON file
        
        Expected JSON format:
        [
            {
                "videoId": "abc123",
                "title": "Video Title",
                "transcript": "Full transcript text...",
                "notes": "Optional: Existing notes/summary..."  # Optional
            },
            ...
        ]
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_transcripts_from_directory(self, directory: str) -> List[Dict]:
        """
        Load transcripts from directory of text files
        
        Directory structure:
        transcripts/
            video1.txt
            video1_notes.txt  # Optional
            video2.txt
            ...
        """
        transcripts = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt') and not filename.endswith('_notes.txt'):
                video_id = filename.replace('.txt', '')
                transcript_path = os.path.join(directory, filename)
                notes_path = os.path.join(directory, f"{video_id}_notes.txt")
                
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                
                notes = None
                if os.path.exists(notes_path):
                    with open(notes_path, 'r', encoding='utf-8') as f:
                        notes = f.read().strip()
                
                transcripts.append({
                    'videoId': video_id,
                    'title': video_id,
                    'transcript': transcript,
                    'notes': notes
                })
        
        return transcripts
    
    def clean_transcript(self, text: str) -> str:
        """Clean and normalize transcript text"""
        # Remove timestamps if present (format: [00:00] or 00:00)
        text = re.sub(r'\[\d{1,2}:\d{2}\]', '', text)
        text = re.sub(r'\d{1,2}:\d{2}', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove speaker labels if present (format: Speaker 1: or [Speaker 1])
        text = re.sub(r'\[?Speaker\s+\d+\]?:?\s*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def generate_notes_from_transcript(self, transcript: str) -> Optional[str]:
        """
        Generate notes from transcript using the trained model
        This creates the target (summary) for training
        """
        try:
            from inference import generate_notes
            notes = generate_notes(transcript, detailed=True, enhanced=False)
            return notes
        except Exception as e:
            print(f"Warning: Could not generate notes: {e}")
            return None
    
    def prepare_training_pairs(
        self, 
        transcripts: List[Dict],
        use_existing_notes: bool = True,
        generate_missing_notes: bool = True,
        min_transcript_length: int = 200,
        max_transcript_length: int = 5000
    ) -> List[Dict]:
        """
        Prepare transcript-note pairs for training
        
        Args:
            transcripts: List of transcript dictionaries
            use_existing_notes: Use existing notes if available
            generate_missing_notes: Generate notes for transcripts without notes
            min_transcript_length: Minimum transcript length (characters)
            max_transcript_length: Maximum transcript length (characters)
        
        Returns:
            List of training pairs in format: {"article": transcript, "abstract": notes}
        """
        training_pairs = []
        
        for item in transcripts:
            transcript = self.clean_transcript(item.get('transcript', ''))
            
            # Filter by length
            if len(transcript) < min_transcript_length:
                print(f"Skipping {item.get('videoId', 'unknown')}: transcript too short")
                continue
            
            if len(transcript) > max_transcript_length:
                # Truncate long transcripts
                transcript = transcript[:max_transcript_length]
                print(f"Truncating {item.get('videoId', 'unknown')}: transcript too long")
            
            # Get or generate notes
            notes = None
            
            if use_existing_notes and item.get('notes'):
                notes = item['notes'].strip()
            
            if not notes and generate_missing_notes:
                print(f"Generating notes for {item.get('videoId', 'unknown')}...")
                notes = self.generate_notes_from_transcript(transcript)
            
            if not notes:
                print(f"Skipping {item.get('videoId', 'unknown')}: no notes available")
                continue
            
            # Validate notes length
            if len(notes) < 50:
                print(f"Skipping {item.get('videoId', 'unknown')}: notes too short")
                continue
            
            training_pairs.append({
                'article': transcript,
                'abstract': notes,
                'videoId': item.get('videoId', 'unknown'),
                'title': item.get('title', 'Unknown')
            })
        
        return training_pairs
    
    def save_training_data(
        self, 
        training_pairs: List[Dict],
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Save training data in Hugging Face datasets format
        
        Args:
            training_pairs: List of training pairs
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set
        """
        # Shuffle data
        import random
        random.shuffle(training_pairs)
        
        # Split data
        total = len(training_pairs)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        train_data = training_pairs[:train_end]
        val_data = training_pairs[train_end:val_end]
        test_data = training_pairs[val_end:]
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        # Save to disk
        output_path = os.path.join(self.output_dir, 'youtube_transcripts_dataset')
        dataset_dict.save_to_disk(output_path)
        
        # Also save as JSON for inspection
        with open(os.path.join(self.output_dir, 'training_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'train': train_data,
                'validation': val_data,
                'test': test_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Training data saved!")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Validation: {len(val_data)} samples")
        print(f"   Test: {len(test_data)} samples")
        print(f"   Total: {total} samples")
        print(f"   Saved to: {output_path}")
        
        return output_path


def main():
    """Main function for preparing training data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YouTube transcript data for training')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file or directory containing transcripts')
    parser.add_argument('--output', type=str, default='./training_data',
                       help='Output directory for training data')
    parser.add_argument('--input-type', type=str, choices=['json', 'directory'], default='json',
                       help='Type of input: json file or directory')
    parser.add_argument('--min-length', type=int, default=200,
                       help='Minimum transcript length (characters)')
    parser.add_argument('--max-length', type=int, default=5000,
                       help='Maximum transcript length (characters)')
    parser.add_argument('--use-existing-notes', action='store_true', default=True,
                       help='Use existing notes if available')
    parser.add_argument('--generate-notes', action='store_true', default=True,
                       help='Generate notes for transcripts without notes')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction for training set')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Fraction for validation set')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = TranscriptDataPreparer(output_dir=args.output)
    
    # Load transcripts
    print(f"📂 Loading transcripts from {args.input}...")
    if args.input_type == 'json':
        transcripts = preparer.load_transcripts_from_json(args.input)
    else:
        transcripts = preparer.load_transcripts_from_directory(args.input)
    
    print(f"✅ Loaded {len(transcripts)} transcripts")
    
    # Prepare training pairs
    print("\n🔄 Preparing training pairs...")
    training_pairs = preparer.prepare_training_pairs(
        transcripts,
        use_existing_notes=args.use_existing_notes,
        generate_missing_notes=args.generate_notes,
        min_transcript_length=args.min_length,
        max_transcript_length=args.max_length
    )
    
    print(f"✅ Prepared {len(training_pairs)} training pairs")
    
    # Save training data
    print("\n💾 Saving training data...")
    output_path = preparer.save_training_data(
        training_pairs,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split
    )
    
    print(f"\n🎉 Done! Training data ready at: {output_path}")
    print(f"\nNext step: Update train.py to use this dataset, or run:")
    print(f"  python train_custom.py --dataset {output_path}")


if __name__ == '__main__':
    main()

