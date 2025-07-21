from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from trl import SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
import os


@dataclass
class LoraConfig:
    """LoRA configuration parameters"""
    finetune_vision_layers: bool = True     # False if not finetuning vision
    finetune_language_layers: bool = True   # False if not finetuning language
    finetune_attention_modules: bool = True  # False if not finetuning attn
    finetune_mlp_modules: bool = True       # False if not finetuning MLP

    r: int = 16             # Higher = more accuracy, might overfit
    lora_alpha: int = 16    # Recommended alpha == r at least
    lora_dropout: float = 0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False    # We support rank stabilized LoRA
    loftq_config = None         # And LoftQ
    # target_modules = "all-linear"  # Optional now! Can specify a list


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    def __post_init__(self):
        # Import here to avoid circular imports
        from constants import model, tokenizer
        from dataset import trainDataSet
        
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = UnslothVisionDataCollator(model, tokenizer)  # Must use!
        self.train_dataset = trainDataSet
    args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        # num_train_epochs=1, # Set this instead of max_steps for full runs
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",      # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    )


@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50


class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
    
    def save_config(self, lora_config: LoraConfig,
                    training_config: TrainingConfig,
                    inference_config: InferenceConfig):
        """Save configuration to file"""
        config_dict = {
            "lora": asdict(lora_config),
            "training": asdict(training_config),
            "inference": asdict(inference_config)
        }
        
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def load_config(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            # Return default configs if file doesn't exist
            return LoraConfig(), TrainingConfig(), InferenceConfig()
        
        import yaml
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        lora_config = LoraConfig(**config_dict.get("lora", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))
        
        return lora_config, training_config, inference_config
