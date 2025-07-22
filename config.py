from dataclasses import dataclass
from trl import SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from dataset import trainDataSet


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    def __post_init__(self):
        # Import here to avoid circular imports
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = UnslothVisionDataCollator(model, tokenizer)
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
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    )


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


@dataclass
class InferenceConfig:
    max_new_tokens: int = 600
    temperature: float = 0.0
    use_cache = True
    min_p = 0.1


inference_config = InferenceConfig()
lora_config = LoraConfig()
