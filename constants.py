from unsloth import FastVisionModel
from config import LoraConfig
from dataclasses import asdict


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

lora_config = LoraConfig()
model = FastVisionModel.get_peft_model(model, **asdict(lora_config))
