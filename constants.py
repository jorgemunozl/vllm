from unsloth import FastVisionModel
from config import LoraConfig
from dataclasses import asdict


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,  # False for 16bit Lora
    use_gradient_checkpointing="unsloth",
)  # Model Base !

# Create an instance of LoraConfig and convert to dict for base model
lora_config = LoraConfig()
model = FastVisionModel.get_peft_model(model, **asdict(lora_config))
instrunctionInference = """Analyze the flowchart image and convert it to Mermaid syntax. Follow these requirements strictly:

1. Use proper Mermaid flowchart syntax starting with 'flowchart TD' (top-down) or 'flowchart LR' (left-right)
2. Identify all nodes/boxes and give them appropriate IDs (A, B, C, etc.)
3. Include all decision diamonds with proper syntax using {condition?}
4. Add all connecting arrows and labels using ONLY '-->' (not '-- >' or other variations)
5. Use appropriate node shapes:
   - [Text] for process boxes
   - {Text} for decision diamonds  
   - ((Text)) for start/end circles
   - [Text] for regular rectangles
6. Include all text labels exactly as shown in the image
7. Ensure proper flow direction and connections

IMPORTANT MERMAID SYNTAX RULES:
- Always use '-->' for arrows (never '-- >')
- Node IDs must be followed immediately by node content: A[Process] or B{Decision?}
- Decision nodes use curly braces: {Is condition met?}
- Start/End nodes use double parentheses: ((Start)) or ((End))
- Process nodes use square brackets: [Do something]
- No spaces in arrow syntax: A --> B (not A -- > B)

EXAMPLE:
```mermaid
flowchart TD
    A((Start)) --> B[Process 1]
    B --> C{Decision?}
    C --> D[Process 2]
    C --> E[Process 3]
    D --> F((End))
    E --> F
```

Provide only the Mermaid code without any additional explanation."""