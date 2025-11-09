"""
Configuration file for ILA Notes Generation Model
"""

# Model Configuration
MODEL_NAME = "facebook/bart-large-cnn"
MODEL_DIR = "./ila-notes-generator"

# Dataset Configuration
DATASET_NAME = "ccdv/arxiv-summarization"
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500

# Tokenization Configuration
MAX_INPUT_LENGTH = 1024  # Maximum tokens for input transcript
MAX_TARGET_LENGTH = 256  # Maximum tokens for generated notes

# Training Configuration
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Generation Configuration
DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_MIN_LENGTH = 50
DEFAULT_TEMPERATURE = 1.0

# Short Notes Configuration
SHORT_NOTES_MAX_LENGTH = 150
SHORT_NOTES_MIN_LENGTH = 30
SHORT_NOTES_NUM_BEAMS = 3
SHORT_NOTES_LENGTH_PENALTY = 1.5

# Detailed Notes Configuration
DETAILED_NOTES_MAX_LENGTH = 256
DETAILED_NOTES_MIN_LENGTH = 100
DETAILED_NOTES_NUM_BEAMS = 5
DETAILED_NOTES_LENGTH_PENALTY = 2.5

# Device Configuration
USE_GPU = True  # Will auto-detect if available
FP16 = True  # Mixed precision training (if GPU available)

