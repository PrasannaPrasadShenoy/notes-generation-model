"""
Train ILA Notes Generation Model on Custom YouTube Transcript Data
Fine-tunes the model on your own transcript data instead of the default dataset
"""

import os
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import json
from config import (
    MODEL_NAME, OUTPUT_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    NUM_EPOCHS, LEARNING_RATE, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, WARMUP_STEPS, WEIGHT_DECAY, USE_GPU, FP16
)

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  No GPU detected. Training will use CPU (slower).")
        return False

def load_custom_dataset(dataset_path: str):
    """Load custom dataset from disk"""
    print(f"\n📚 Loading custom dataset from {dataset_path}...")
    
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        
        if isinstance(dataset, DatasetDict):
            print(f"   Train size: {len(dataset['train'])}")
            print(f"   Validation size: {len(dataset.get('validation', []))}")
            if 'test' in dataset:
                print(f"   Test size: {len(dataset['test'])}")
        else:
            print(f"   Dataset size: {len(dataset)}")
        
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def preprocess_function(examples, tokenizer):
    """Preprocess the dataset for training"""
    # Map article to inputs and abstract to targets
    inputs = examples["article"]
    targets = examples["abstract"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss function
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train notes generation model on custom data')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to custom dataset (from prepare_training_data.py)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for trained model (default: ./ila-notes-generator-custom)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 ILA Notes Generation Model - Custom Training")
    print("=" * 60)
    
    # Check GPU
    use_gpu = check_gpu()
    
    # Set output directory
    output_dir = args.output or "./ila-notes-generator-custom"
    
    # Load custom dataset
    dataset = load_custom_dataset(args.dataset)
    
    # Handle DatasetDict vs single Dataset
    if isinstance(dataset, DatasetDict):
        train_data = dataset['train']
        val_data = dataset.get('validation', dataset.get('val', None))
        if val_data is None and 'test' in dataset:
            # Use test set as validation if no validation set
            val_data = dataset['test']
    else:
        # Single dataset - split it
        split = dataset.train_test_split(test_size=0.2)
        train_data = split['train']
        val_data = split['test']
    
    if val_data is None:
        print("⚠️  No validation set found. Using 20% of training data for validation.")
        split = train_data.train_test_split(test_size=0.2)
        train_data = split['train']
        val_data = split['test']
    
    print(f"\n📊 Training configuration:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print("✅ Model and tokenizer loaded!")
    
    # Preprocess datasets
    print("\n🔄 Preprocessing datasets...")
    print("   This may take a few minutes...")
    
    train_tokenized = train_data.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing training data"
    )
    
    val_tokenized = val_data.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizing validation data"
    )
    
    print("✅ Preprocessing complete!")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments (use command line args if provided)
    num_epochs = args.epochs or NUM_EPOCHS
    learning_rate = args.learning_rate or LEARNING_RATE
    batch_size = args.batch_size or TRAIN_BATCH_SIZE
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size if use_gpu else 1,
        per_device_eval_batch_size=EVAL_BATCH_SIZE if use_gpu else 1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=learning_rate,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",  # Changed from evaluation_strategy in newer transformers
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=FP16 and use_gpu,
        dataloader_num_workers=2 if use_gpu else 0,
        report_to="none",
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective batch size: {batch_size * GRADIENT_ACCUMULATION_STEPS}")
    
    train_result = trainer.train()
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "dataset": args.dataset,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "training_loss": train_result.training_loss,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "custom_training": True
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Training loss: {train_result.training_loss:.4f}")
    
    # Test inference
    print("\n🧪 Testing model with sample input...")
    sample_text = train_data[0]["article"][:500] if len(train_data) > 0 else "Sample text for testing..."
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    
    if use_gpu:
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    generated_notes = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"\n📝 Sample Generated Notes:")
    print("-" * 60)
    print(generated_notes)
    print("-" * 60)
    
    print("\n🎉 All done! You can now use this model in inference.py")
    print(f"   Update MODEL_DIR in config.py to: {output_dir}")

if __name__ == "__main__":
    main()

