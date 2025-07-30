---
model:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_seq_length: 2048
  use_4bit: true
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
training:
  dataset_name: "alpaca"
  batch_size: 4
  learning_rate: 2.0e-4
  num_epochs: 3
---

# Flowchart-to-Mermaid Transformer

This project converts flowchart images into [Mermaid](https://mermaid-js.github.io/) code using a vision-language model (Llama 3.2 Vision + LoRA adapters). It supports both fine-tuned (FT) and non-fine-tuned (NFT) model evaluation.


## 🚀 Quick Start

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run inference:**
    ```bash
    python inferenceNFT.py
    # or for FT model
    python inferenceFT.py
    ```

3. **Evaluate and compare:**
    ```bash
    python metric.py
    python plot_comparison.py
    ```

---

## 📊 Model Performance

| Metric                | NFT Model | FT Model | Improved Generator |
|-----------------------|-----------|----------|-------------------|
| Levenshtein Similarity|   44.3%   |  63.6%   |      70.1%        |
| Jaccard Similarity    |   52.7%   |  68.2%   |      74.5%        |
| Cosine Similarity     |   60.1%   |  75.3%   |      80.2%        |
| Sequence Matcher      |   47.8%   |  66.7%   |      72.9%        |
| Hamming Similarity    |   10.2%   |  15.4%   |      18.7%        |
| Jaro Similarity       |   55.6%   |  70.8%   |      77.0%        |

*These are average scores over the test set. See `metric.py` for details.*

---

## 🖼️ Example Output

**Input:**
![example](example_image.png)

**Generated Mermaid:**
```
flowchart TD
    A((Start)) --> B["Load Application"]
    B --> C[/"User Input Required"/]
    C --> D{"Valid Input?"}
    D -->|Yes| E["Process Request"]
    D -->|No| F["Show Error Message"]
    E --> G[/"Display Results"/]
    F --> C
    G --> H((End))
```

---

## 🛠️ Project Structure

- `inferenceNFT.py` / `inferenceFT.py` — Run inference with NFT/FT models
- `metric.py` — String similarity metrics
- `plot_comparison.py` — Visualize and compare results
- `utilsFunctions.py` — Utilities for cleaning and formatting
- `NFT/`, `FT/`, `groundT/` — Output and reference directories

---

## 📦 Model Weights

- **LoRA adapters** are available on [Hugging Face](https://huggingface.co/your-username/your-model-repo).
- To use, download the safetensors and config files and place them in your project directory.

---

## 📑 Citation

If you use this project, please cite:
```
@misc{flowchart2mermaid2025,
  author = {Your Name},
  title = {Flowchart-to-Mermaid Transformer},
  year = {2025},
  url = {https://github.com/your-username/your-repo}
}
```
  output_dir: "./results"
  save_strategy: "epoch"
  logging_steps: 1

inference:
  max_new_tokens: 256
  temperature: 0.7
  do_sample: true
  top_p: 0.9
  top_k: 50
---


## VLLMS TRAINED ON MERMAID

Use -> MananSuri27/Flowchart2Mermaid