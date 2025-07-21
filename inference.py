from transformers import TextStreamer
from constants import model, tokenizer
from dataset import testDataSet
from constants import instrunctionInference

# Extract the PIL Image from the dataset dictionary
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
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
response = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500,
                          use_cache=True, temperature=1.5, min_p=0.1)
print(response)
