import os
import sys
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the training/inference pipeline"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    dataset_name: str = "alpaca"
    max_seq_length: int = 2048
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    output_dir: str = "./results"
    use_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class ModelManager:
    """Manages model loading and configuration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Try loading with Unsloth (if available)
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=self.config.use_4bit,
            )
            
            if self.config.use_lora:
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.config.lora_r,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )
            
        except ImportError:
            logger.warning("Unsloth not available, using standard transformers")
            # Fallback to standard transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def load_dataset(self):
        """Load and preprocess the dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Example for Alpaca dataset
        if self.config.dataset_name == "alpaca":
            dataset = load_dataset("tatsu-lab/alpaca")
        else:
            # Add your custom dataset loading logic here
            dataset = load_dataset(self.config.dataset_name)
        
        # Preprocessing function
        def format_prompts(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            
            texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                prompt = f"### Instruction:\n{instruction}\n"
                if input_text:
                    prompt += f"### Input:\n{input_text}\n"
                prompt += f"### Response:\n{output}"
                texts.append(prompt)
            
            return {"text": texts}
        
        dataset = dataset.map(format_prompts, batched=True)
        logger.info(f"Dataset loaded with {len(dataset['train'])} samples")
        return dataset

def train_model(config: Config):
    """Main training function"""
    logger.info("Starting training pipeline")
    
    # Initialize model manager
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model()
    
    # Load and process data
    data_processor = DataProcessor(config, tokenizer)
    dataset = data_processor.load_dataset()
    
    # Training setup (using Unsloth or transformers)
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=5,
                num_train_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=config.output_dir,
                save_strategy="epoch",
            ),
        )
        
        # Start training
        logger.info("Beginning training...")
        trainer.train()
        
        # Save the model
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        
    except ImportError as e:
        logger.error(f"Training dependencies not available: {e}")
        logger.info("Please install required packages: pip install trl transformers")

def inference_example(config: Config, prompt: str):
    """Example inference function"""
    logger.info("Running inference example")
    
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model()
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated response: {response}")
    return response

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Choose mode: train or inference
    mode = input("Choose mode (train/inference): ").strip().lower()
    
    if mode == "train":
        train_model(config)
    elif mode == "inference":
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            prompt = "### Instruction:\nExplain what is machine learning.\n### Response:\n"
        inference_example(config, prompt)
    else:
        logger.error("Invalid mode. Please choose 'train' or 'inference'")
        sys.exit(1)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()
