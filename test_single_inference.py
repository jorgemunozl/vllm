from transformers import TextStreamer
from constants import model, tokenizer
from dataset import testDataSet
from constants import instrunctionInference

# Extract the first PIL Image from the dataset dictionary
image = testDataSet[0]["image"]  # This is a PIL.Image object

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

# Generate response
response = model.generate(**inputs, max_new_tokens=500,
                          use_cache=True, temperature=1.0, min_p=0.1)

# Decode the response to get the generated text
generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

# Extract only the assistant's response
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

# Save to test file
with open("test_output.mmd", 'w', encoding='utf-8') as f:
    f.write(assistant_response)

print("="*60)
print("IMPROVED MERMAID OUTPUT:")
print("="*60)
print(assistant_response)
print("="*60)
print("Saved to: test_output.mmd")
