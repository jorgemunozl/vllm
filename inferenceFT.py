from unsloth import FastVisionModel
from utilsFunctions import generateMermaids
from dataset import testDataSet
from peft import PeftModel  # Import for PEFT adapter loading
import torch  # for clearing GPU cache
import os


def clear_cuda():
    """Clear all CUDA memory and variables"""
    # Delete any existing models/variables if they exist
    import gc
    if 'model' in globals():
        del globals()['model']
    if 'tokenizer' in globals():
        del globals()['tokenizer']
    # Run garbage collection
    gc.collect()
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Cleared all CUDA memory")

# Clear CUDA memory before starting
clear_cuda()

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)


model = PeftModel.from_pretrained(model, "jorgemunozl/flowchart2mermaid")
model = model.merge_and_unload()
torch.cuda.empty_cache()

if __name__ == "__main__":
    generateMermaids(testDataSet, tokenizer, "output", model, 4, 5)