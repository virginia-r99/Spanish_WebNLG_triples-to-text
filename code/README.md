# Data-to-Text Evaluation for Spanish and English WebNLG

This folder contains code for evaluating data-to-text generation in Spanish and English using WebNLG datasets. The structure of the repository is as follows:

## Folder Structure

- **context_learning_evaluation/**  
  Contains code for evaluating HuggingFace instruct models:
  - `verb_evaluation_CausalLM_WebNLG_ES_instruct.py` (for Spanish)
  - `verb_evaluation_CausalLM_WebNLG_ES_instruct_English.py` (for English)

- **fine_tuning/**  
  Contains scripts for fine-tuning each model using LoRA.

- **fine_tuning_evaluation/**  
  Contains adaptations of the evaluation scripts from `context_learning_evaluation` to assess the performance of the fine-tuned models.

- **metrics_computation/**

  Contain the metric computation codes:
  - `compute_metrics_instruct.py` → Computes the metrics for each evaluation entry in a folder of JSON files.
  - `test_results_instruct.ipynb` → Computes the average metrics for each JSON file.

## Usage

### Context Learning Evaluation
Run the appropriate script to evaluate the instruct models:
```bash
python context_learning_evaluation/verb_evaluation_CausalLM_WebNLG_ES_instruct.py  # Spanish evaluation
python context_learning_evaluation/verb_evaluation_CausalLM_WebNLG_ES_instruct_English.py  # English evaluation
```

### Fine-Tuning
Fine-tune the models using LoRA by running the scripts in `fine_tuning/`.

### Fine-Tuned Model Evaluation
Evaluate the fine-tuned models using the adapted scripts in `fine_tuning_evaluation/`.

## Requirements
Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

