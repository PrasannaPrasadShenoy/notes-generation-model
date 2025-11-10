"""
LoRA Fine-tuning Module
Efficiently fine-tune LLMs using LoRA (Low-Rank Adaptation)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import json
from typing import Optional
from config import (
    MODEL_NAME, OUTPUT_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    NUM_EPOCHS, LEARNING_RATE, TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
)


class LoRATrainer:
    """Train models using LoRA for efficient fine-tuning"""
    
    def __init__(
        self,
        base_model_name: str = "google/flan-t5-large",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ):
        """
        Initialize LoRA trainer
        
        Args:
            base_model_name: Base model to fine-tune
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            target_modules: Modules to apply LoRA (auto-detected if None)
        """
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.model = None
        self.tokenizer = None
        self.is_seq2seq = "t5" in base_model_name.lower() or "bart" in base_model_name.lower()
    
    def setup_model(self):
        """Load and configure model with LoRA"""
        print(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # Configure LoRA
        if self.target_modules is None:
            # Auto-detect target modules
            if "t5" in self.base_model_name.lower():
                self.target_modules = ["q", "v", "k", "o"]
            elif "bart" in self.base_model_name.lower():
                self.target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
            else:
                self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        task_type = TaskType.SEQ_2_SEQ_LM if self.is_seq2seq else TaskType.CAUSAL_LM
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=task_type
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Model setup complete with LoRA")
    
    def prepare_dataset(
        self,
        dataset_name: Optional[str] = None,
        instruction_dataset: Optional[str] = "databricks/databricks-dolly-15k",
        max_samples: int = 1000
    ):
        """
        Prepare training dataset
        
        Args:
            dataset_name: Custom dataset name
            instruction_dataset: Instruction dataset for fine-tuning
            max_samples: Maximum samples to use
        """
        datasets = []
        
        # Load instruction dataset
        if instruction_dataset:
            print(f"Loading instruction dataset: {instruction_dataset}")
            try:
                inst_data = load_dataset(instruction_dataset, split=f"train[:{max_samples}]")
                datasets.append(inst_data)
            except Exception as e:
                print(f"Warning: Could not load {instruction_dataset}: {e}")
        
        # Load summarization dataset
        print("Loading summarization dataset: ccdv/arxiv-summarization")
        try:
            summ_data = load_dataset("ccdv/arxiv-summarization", split=f"train[:{max_samples//2}]")
            datasets.append(summ_data)
        except Exception as e:
            print(f"Warning: Could not load summarization dataset: {e}")
        
        # Combine datasets
        if datasets:
            from datasets import concatenate_datasets
            combined = concatenate_datasets(datasets)
        else:
            raise ValueError("No datasets loaded")
        
        # Preprocess
        def preprocess_function(examples):
            if self.is_seq2seq:
                # For seq2seq models
                if "article" in examples and "abstract" in examples:
                    inputs = examples["article"]
                    targets = examples["abstract"]
                elif "instruction" in examples and "response" in examples:
                    inputs = [f"{inst}\n{context}" if context else inst 
                             for inst, context in zip(examples["instruction"], examples.get("context", [""]*len(examples["instruction"])))]
                    targets = examples["response"]
                else:
                    # Fallback
                    inputs = examples.get("input", examples.get("text", [""]))
                    targets = examples.get("output", examples.get("summary", [""]))
                
                model_inputs = self.tokenizer(
                    inputs,
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    padding="max_length"
                )
                
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets,
                        max_length=MAX_TARGET_LENGTH,
                        truncation=True,
                        padding="max_length"
                    )
                
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            else:
                # For causal LM
                texts = [f"{inst}\n{resp}" for inst, resp in 
                        zip(examples.get("instruction", [""]), examples.get("response", [""]))]
                return self.tokenizer(
                    texts,
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    padding="max_length"
                )
        
        tokenized = combined.map(
            preprocess_function,
            batched=True,
            remove_columns=combined.column_names
        )
        
        return tokenized
    
    def train(
        self,
        train_dataset,
        output_dir: str = "./ila-insight-lora",
        num_epochs: int = 2,
        learning_rate: float = 2e-4,
        batch_size: int = 4
    ):
        """
        Train the model with LoRA
        
        Args:
            train_dataset: Tokenized training dataset
            output_dir: Output directory
            num_epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        if self.model is None:
            self.setup_model()
        
        # Data collator
        if self.is_seq2seq:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="no",
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("🚀 Starting LoRA training...")
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training complete! Model saved to {output_dir}")


def main():
    """Example training"""
    trainer = LoRATrainer(
        base_model_name="google/flan-t5-large",
        lora_r=8,
        lora_alpha=16
    )
    
    trainer.setup_model()
    dataset = trainer.prepare_dataset(max_samples=500)
    trainer.train(dataset, num_epochs=1)


if __name__ == "__main__":
    main()

