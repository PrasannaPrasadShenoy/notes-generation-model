"""
Example usage of ILA Notes Generation Model
Demonstrates how to use the model to generate notes from transcripts
"""

from inference import NotesGenerator, generate_notes

# Example 1: Simple usage with convenience function
print("=" * 60)
print("Example 1: Simple Usage")
print("=" * 60)

transcript = """
Machine learning is a subset of artificial intelligence that enables computers to learn 
and make decisions from data without being explicitly programmed. It uses algorithms to 
identify patterns in data and make predictions or classifications. There are three main 
types of machine learning: supervised learning, where models learn from labeled data; 
unsupervised learning, which finds patterns in unlabeled data; and reinforcement learning, 
where agents learn through trial and error with rewards and penalties. Deep learning, a 
subset of machine learning, uses neural networks with multiple layers to process complex 
data. Applications of machine learning include image recognition, natural language processing, 
recommendation systems, and autonomous vehicles. The field continues to evolve with advances 
in algorithms, computing power, and data availability.
"""

# Generate short notes
short_notes = generate_notes(transcript, detailed=False)
print("\n📝 Short Notes:")
print(short_notes)

# Generate detailed notes
detailed_notes = generate_notes(transcript, detailed=True)
print("\n📚 Detailed Notes:")
print(detailed_notes)

# Generate enhanced notes (with metadata)
enhanced_notes = generate_notes(transcript, enhanced=True)
print("\n✨ Enhanced Notes:")
print(enhanced_notes)

# Example 2: Using NotesGenerator class for more control
print("\n" + "=" * 60)
print("Example 2: Advanced Usage with NotesGenerator")
print("=" * 60)

generator = NotesGenerator()

# Custom generation parameters
custom_notes = generator.generate_notes(
    transcript,
    max_length=200,
    min_length=80,
    num_beams=5,
    length_penalty=2.5
)
print("\n🎯 Custom Notes:")
print(custom_notes)

# Example 3: Processing multiple transcripts
print("\n" + "=" * 60)
print("Example 3: Batch Processing")
print("=" * 60)

transcripts = [
    "First transcript about neural networks and deep learning...",
    "Second transcript about data science and analytics...",
    "Third transcript about cloud computing and infrastructure..."
]

# Note: In production, you might want to process these in batches
# For now, we'll just show the pattern
print(f"\n📊 Processing {len(transcripts)} transcripts...")
for i, transcript in enumerate(transcripts, 1):
    if len(transcript) > 100:  # Only process if transcript is long enough
        notes = generate_notes(transcript, detailed=True)
        print(f"\nTranscript {i} Notes:")
        print(notes[:200] + "..." if len(notes) > 200 else notes)

print("\n✅ Examples complete!")

