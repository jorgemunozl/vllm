from unsloth import FastVisionModel
from peft import PeftModel
from datasets import load_dataset
import torch
import os
import gc


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


instruction = """
Analyze the flowchart image and convert it to Mermaid syntax. Follow these requirements strictly:
1. Use proper Mermaid flowchart syntax starting with 'flowchart TD' (top-down) or 'flowchart LR' (left-right)
2. Identify all nodes/boxes and give them appropriate IDs (A, B, C, etc.)
3. Include all decision diamonds with proper syntax using {condition?}
4. Add all connecting arrows and labels using ONLY '-->' (not '-- >' or other variations)
5. Use appropriate node shapes:
   - [Text] for process boxes
   - {Text} for decision diamonds
   - ((Text)) for start/end circles
   - [Text] for regular rectangles
   - [/"Text"/] for data input/output parallelograms
6. Include all text labels exactly as shown in the image
7. Ensure proper flow direction and connections
8. Decision branches: `-->|Yes|` and `-->|No|` (exact capitalization)
9. Alternative labels: `-->|True|`, `-->|False|` when appropriate

IMPORTANT MERMAID SYNTAX RULES:
- Always use '-->' for arrows (never '-- >')
- Node IDs must be followed immediately by node content: A["Process"] or B{"Decision?"}
- Decision nodes use curly braces: {"Is condition met?"}
- Start/End nodes use double parentheses: ((Start)) or ((End))
- Process nodes use square brackets: ["Do something"]
- No spaces in arrow syntax: A --> B (not A -- > B)
- Always use quoting to write text. A["Process"]
- The end is always follow by a ID. K((End))

EXAMPLE:
flowchart TD
    A((Start)) --> B["Load Application"]
    B --> C[/"User Input Required"/]
    C --> D{"Valid Input?"}
    D -->|Yes| E["Process Request"]
    D -->|No| F["Show Error Message"]
    E --> G[/"Display Results"/]
    F --> C
    G --> H((End))

Provide only the Mermaid code without any additional explanation."""


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = PeftModel.from_pretrained(model, "jorgemunozl/flowchart2mermaid")
model = model.merge_and_unload()

ds = load_dataset("sroecker/mermaid-flowchart-transformer-moondream-caption")
total_time=0
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
FastVisionModel.for_inference(model)
for i in range(1,60):
    print(f" -> Image {i}")
    image = ds["train"][i]["image"]
    inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
    ).to("cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    res = model.generate(**inputs, max_new_tokens = 2000,
                   use_cache = False, temperature = 0.1, min_p = 0.1)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    total_time += elapsed_time_ms / 1000
    print("Elapsed_time: ",elapsed_time_ms / 1000)
    generated_text = tokenizer.decode(res[0],skip_special_tokens=True)
    # Extract only the assistant's response (everything after the last assistant token)
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    elif "assistant" in generated_text:
        assistant_response = generated_text.split("assistant")[-1]
    else:
        assistant_response = generated_text
    assistant_response = assistant_response.strip()
    if assistant_response.endswith("<|eot_id|>"):
        assistant_response = assistant_response[:-9].strip()
    print(assistant_response)
    os.makedirs("FTBATCH",exist_ok=True)
    with open(f"FTBATCH/{i}.md","w") as f:
        f.write(assistant_response)
    del inputs, res
    torch.cuda.empty_cache()
    gc.collect()
