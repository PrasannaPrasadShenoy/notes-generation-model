"""
ILA Notes Generation Model - Training Script
Fine-tunes a BART model on educational summarization dataset to generate enhanced notes
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from tqdm import tqdm
import json
from config import (
    MODEL_NAME, DATASET_NAME, OUTPUT_DIR, TRAIN_SAMPLES, VAL_SAMPLES,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, NUM_EPOCHS, LEARNING_RATE,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS, WEIGHT_DECAY, USE_GPU, FP16
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

def load_and_prepare_dataset():
    """Load and prepare the dataset"""
    print("\n📚 Loading dataset...")
    print(f"   Dataset: {DATASET_NAME}")
    
    try:
        dataset = load_dataset(DATASET_NAME)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train size: {len(dataset['train'])}")
        print(f"   Validation size: {len(dataset['validation'])}")
        
        # Select subset for faster training
        train_data = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
        val_data = dataset["validation"].select(range(min(VAL_SAMPLES, len(dataset["validation"]))))
        
        print(f"\n📊 Using subset:")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        return train_data, val_data
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
    print("=" * 60)
    print("🧠 ILA Notes Generation Model - Training")
    print("=" * 60)
    
    # Check GPU
    use_gpu = check_gpu()
    
    # Load dataset
    train_data, val_data = load_and_prepare_dataset()
    
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE if use_gpu else 1,
        per_device_eval_batch_size=EVAL_BATCH_SIZE if use_gpu else 1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=FP16 and use_gpu,  # Use mixed precision if GPU available
        dataloader_num_workers=2 if use_gpu else 0,
        report_to="none",  # Disable wandb/tensorboard
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
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    train_result = trainer.train()
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "training_loss": train_result.training_loss,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
    }
    
    with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Training loss: {train_result.training_loss:.4f}")
    
    # Test inference
    print("\n🧪 Testing model with sample input...")
    sample_text = train_data[0]["article"][:500]  # First 500 chars
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
    
    print("\n🎉 All done! You can now use inference.py to generate notes.")

if __name__ == "__main__":
    main()

