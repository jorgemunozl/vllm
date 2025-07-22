from transformers import TextStreamer
from constants import model, tokenizer
from dataset import testDataSet
from constants import instrunctionInference
import os
import torch
import gc

os.makedirs("mermaid_outputs", exist_ok=True)

torch.cuda.empty_cache() # Relevant to window context? 
gc.collect()

start_index = 30  
end_index = 70   

num_images = min(end_index, len(testDataSet)) - start_index
print(f"Processing images {start_index + 1} to {min(end_index, len(testDataSet))} from the dataset...")

for i in range(start_index, min(end_index, len(testDataSet))):
    current_image_num = i + 1
    print(f"\n{'='*60}")
    print(f"Processing image {current_image_num}/{len(testDataSet)} (batch: {current_image_num - start_index}/{num_images})")
    print('='*60)

    # Clear cache before each inference <-----
    torch.cuda.empty_cache()
    # Extract the PIL Image from the dataset dictionary
    image = testDataSet[i]["image"]  # This is a PIL.Image object
    instruction = instrunctionInference
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate response with reduced temperature for better consistency
    response = model.generate(**inputs, max_new_tokens=500,
                              use_cache=True, temperature=1.0, min_p=0.1)
    
    # Decode the response to get the generated text
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (everything after the last assistant token)
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    elif "assistant" in generated_text:
        assistant_response = generated_text.split("assistant")[-1]
    else:
        assistant_response = generated_text
    
    # Clean up the response
    assistant_response = assistant_response.strip()
    if assistant_response.endswith("<|eot_id|>"):
        assistant_response = assistant_response[:-9].strip()
    
    # Save to .mmd file
    output_filename = f"mermaid_outputs/flowchart_{current_image_num:02d}.mmd"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(assistant_response)
    
    print(f"Generated Mermaid code saved to: {output_filename}")
    print(f"Preview of generated content:")
    print("-" * 40)
    # Show first few lines of the output
    preview_lines = assistant_response.split('\n')[:10]
    for line in preview_lines:
        print(line)
    if len(assistant_response.split('\n')) > 10:
        print("... (truncated)")
    print("-" * 40)
    
    # Clean up tensors
    del inputs, response
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*60}")
print(f"✅ Successfully processed images {start_index + 1} to {min(end_index, len(testDataSet))}!")
print(f"📁 All Mermaid files saved in 'mermaid_outputs/' directory")
print('='*60)
