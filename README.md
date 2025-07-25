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
# Model Performance Metrics

This section details the performance of the vision-to-text model in generating Mermaid diagrams from flowchart images. We compare three versions:

1.  **NFT (No Fine-Tuning):** The base model without any fine-tuning.
2.  **FT (Fine-Tuned):** The model after being fine-tuned on the `MananSuri27/Flowchart2Mermaid` dataset.
3.  **Improved:** The output from the fine-tuned model after applying a rule-based post-processing script (`utilsFunctions.post_process_mermaid`).

## Comparison Results

The following table and chart summarize the average similarity scores between the generated Mermaid diagrams and the ground truth.

| Metric                      | NFT (No Fine-Tuning) | FT (Fine-Tuned) | Improved (Post-Processed) | Best Model |
| --------------------------- | -------------------- | --------------- | ------------------------- | ---------- |
| Levenshtein Similarity      | 58.62                | 53.51           | 58.62                     | NFT        |
| Jaccard Similarity          | 89.30                | 88.28           | 89.30                     | NFT        |
| Cosine Similarity           | 96.55                | 96.19           | 96.55                     | NFT        |
| Sequence Matcher Similarity | 70.88                | 66.83           | 70.88                     | NFT        |
| Hamming Similarity          | 70.88                | 66.83           | 70.88                     | NFT        |
| Jaro Similarity             | 84.88                | 83.17           | 84.88                     | NFT        |
| **OVERALL AVERAGE**         | **71.85**            | **69.14**       | **71.85**                 | **NFT**    |

![Model Comparison](comparison_with_improved.png)

## Key Observations

-   The **No Fine-Tuning (NFT)** model and the **Improved** model achieved the same overall average similarity of **71.85%**.
-   The **Fine-Tuned (FT)** model performed slightly worse, with an average similarity of **69.14%**. This suggests that the fine-tuning process may have introduced some regressions, or that the base model was already quite capable.
-   The post-processing script was effective in correcting many of the syntax errors from the FT model, bringing its performance back up to the level of the NFT model.

Further analysis is needed to understand the specific strengths and weaknesses of each model version.
