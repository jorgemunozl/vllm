# Flowchart to Mermaid - Fine-tuned Llama 3.2 Vision

This is a fine-tuned version of Llama 3.2 11B Vision model specifically trained to convert flowchart images into Mermaid syntax.

## Model Description

- **Base Model**: unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Vision-to-text conversion (Flowchart → Mermaid)
- **Dataset**: MananSuri27/Flowchart2Mermaid

## Usage

```python
from unsloth import FastVisionModel
from PIL import Image

# Load the model
model, tokenizer = FastVisionModel.from_pretrained(
    "your-username/flowchart-to-mermaid-llama",  # Replace with your model name
    load_in_4bit=True,
)

# Load your flowchart image
image = Image.open("flowchart.png")

# Create messages
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Convert this flowchart to Mermaid syntax"}
    ]}
]

# Generate
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=500)
```

## Training Details

- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Target Modules**: Vision and language projection layers
- **Quantization**: 4-bit

## Files

- `adapter_model.safetensors`: Main LoRA adapter weights
- `adapter_config.json`: LoRA configuration
- `config.yaml`: Training configuration

## Citation

If you use this model, please cite the original Llama 3.2 paper and the dataset:

```bibtex
@misc{flowchart2mermaid,
  title={Flowchart to Mermaid Fine-tuned Model},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/your-username/flowchart-to-mermaid-llama}
}
```
