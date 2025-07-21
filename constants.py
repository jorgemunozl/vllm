from unsloth import FastVisionModel
from config import LoraConfig


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,  # False for 16bit Lora
    use_gradient_checkpointing="unsloth",
)  # Model Base !

model = FastVisionModel.get_peft_model(model, LoraConfig)
instrunctionInference = "prom"
