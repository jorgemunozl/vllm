from unsloth import FastVisionModel
from config import LoraConfig
from dataclasses import asdict


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,  # False for 16bit Lora
    use_gradient_checkpointing="unsloth",
)  # Model Base !

# Load the fine-tuned adapter from local path
# Replace "./path-to-your-adapter" with the actual path to your adapter files
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="./path-to-your-adapter",  # Path where your adapter_config.json and .safetensors are located
    model=model,
    tokenizer=tokenizer,
    load_in_4bit=True,
)
instrunctionInference = """Analyze the flowchart image and convert it to Mermaid syntax. Follow these requirements:

1. Use proper Mermaid flowchart syntax starting with 'flowchart TD' (top-down) or 'flowchart LR' (left-right)
2. Identify all nodes/boxes and give them appropriate IDs (A, B, C, etc.)
3. Include all decision diamonds with proper syntax using {condition?}
4. Add all connecting arrows and labels
5. Use appropriate node shapes:
   - [ ] for process boxes
   - { } for decision diamonds  
   - (( )) for start/end circles
   - [ ] for regular rectangles
6. Include all text labels exactly as shown in the image
7. Ensure proper flow direction and connections

Provide only the Mermaid code without any additional explanation."""