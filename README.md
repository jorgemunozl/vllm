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