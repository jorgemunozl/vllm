import os
from config import inference_config
from dataclasses import asdict
from PIL import Image
import torch  # for clearing GPU cache if needed

with open("instructions.md", "r") as f:
    instruc = f.read()


def generateMermaids(dataset, tokenizer, nameDir,
                     model, start_index=0, end_index=None):
    os.makedirs(nameDir, exist_ok=True)
    # Default to full dataset if end_index not specified
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        # Clear GPU cache per iteration to free memory
        torch.cuda.empty_cache()
        print(f"Number {i}")
        image = dataset[i]["image"]  # This is a PIL.Image object
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruc}
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
        
        # Ensure all tensors are on CUDA
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to("cuda")
        
        response = model.generate(**inputs, **asdict(inference_config))
        generated_text = tokenizer.decode(response[0],
                                          skip_special_tokens=True)
        # Save output under the specified directory
        output_filename = os.path.join(nameDir, f"{i}.md")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(generated_text)


def downloadImagesFromDS(DataSet, index):
    element = DataSet[index]
    print(f"\nDownloading {index} element")
    os.makedirs("downloads", exist_ok=True)
    for key, value in element.items():
        if isinstance(value, Image.Image):
            filename = f"downloads/{key}.png"
            value.save(filename)
        elif isinstance(value, str) and len(value) < 500:  # Text/caption
            filename = f"downloads/first_sample_{key}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(value)
            print(f"✅ Saved text '{key}' to: {filename}")
        elif isinstance(value, (list, dict)):
            filename = f"downloads/first_sample_{key}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(value))
            print(f"✅ Saved data '{key}' to: {filename}")
