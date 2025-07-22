from transformers import TextStreamer
from dataset import trainDataSet
import torch
from utilsFunctions import generateMermaids


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

if __name__ == "__main__":
    generateMermaids(trainDataSet,to)
