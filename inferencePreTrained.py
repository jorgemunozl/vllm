from unsloth import FastVisionModel
from utilsFunctions import generateMermaids
from dataset import trainDataSet

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    adapter_model_name="jorgemunozl/flowchart2mermaid",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

if __name__ == "__main__":
    generateMermaids(trainDataSet, model, tokenizer)